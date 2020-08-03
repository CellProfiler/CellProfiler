import time

import numpy
import wx


class ProgressWatcher:
    """ Tracks pipeline progress and estimates time to completion """

    def __init__(self, parent, update_callback, multiprocessing=False):
        self.update_callback = update_callback
        # start tracking progress
        self.start_time = time.time()
        self.end_times = None
        self.current_module_name = ""
        self.pause_start_time = None
        self.previous_pauses_duration = 0.0
        self.image_set_index = 0
        self.num_image_sets = 1

        # for multiprocessing computation
        self.num_jobs = 1
        self.num_received = 0

        self.multiprocessing = multiprocessing

        timer_id = wx.NewId()
        self.timer = wx.Timer(parent, timer_id)
        self.timer.Start(500)
        if not multiprocessing:
            parent.Bind(wx.EVT_TIMER, self.update, id=timer_id)
            self.update()
        else:
            parent.Bind(wx.EVT_TIMER, self.update_multiprocessing, id=timer_id)
            self.update_multiprocessing()

    def stop(self):
        self.timer.Stop()

    def update(self, event=None):
        status = "%s, Image Set %d/%d" % (
            self.current_module_name,
            self.image_set_index + 1,
            self.num_image_sets,
        )
        self.update_callback(status, self.elapsed_time(), self.remaining_time())

    def update_multiprocessing(self, event=None):
        if self.num_jobs > self.num_received:
            status = "Processing: %d of %d image sets completed" % (
                self.num_received,
                self.num_jobs,
            )
            self.update_callback(
                status, self.elapsed_time(), self.remaining_time_multiprocessing()
            )
        else:
            status = "Post-processing, please wait"
            self.update_callback(status, self.elapsed_time())

    def on_pipeline_progress(self, *args):
        if not self.multiprocessing:
            self.on_start_module(*args)
        else:
            self.on_receive_work(*args)

    def on_start_module(self, module, num_modules, image_set_index, num_image_sets):
        """
        Update the historical execution times, which are used as the
        bases for projecting the time that remains.  Also update the
        labels that show the current module and image set.  This
        method is called by the pipelinecontroller at the beginning of
        every module execution to update the progress bar.
        """
        self.current_module = module
        self.current_module_name = module.module_name
        self.num_modules = num_modules
        self.image_set_index = image_set_index
        self.num_image_sets = num_image_sets

        if self.end_times is None:
            # One extra element at the beginning for the start time
            self.end_times = numpy.zeros(1 + num_modules * num_image_sets)
        module_index = module.module_num - 1  # make it zero-based
        index = image_set_index * num_modules + (module_index - 1)
        self.end_times[1 + index] = self.elapsed_time()

        self.update()

    def on_receive_work(self, num_jobs, num_received):
        self.num_jobs = num_jobs
        self.num_received = num_received
        if self.end_times is None:
            # One extra element at the beginning for the start time
            self.end_times = numpy.zeros(1 + num_jobs)
            self.end_times[0] = self.elapsed_time()
        self.end_times[num_received] = self.elapsed_time()
        self.update_multiprocessing()

    def pause(self, do_pause):
        if do_pause:
            self.pause_start_time = time.time()
        else:
            self.previous_pauses_duration += time.time() - self.pause_start_time
            self.pause_start_time = None

    def adjusted_time(self):
        """Current time minus the duration spent in pauses."""
        pauses_duration = self.previous_pauses_duration
        if self.pause_start_time:
            pauses_duration += time.time() - self.pause_start_time
        return time.time() - pauses_duration

    def elapsed_time(self):
        """Return the number of seconds that have elapsed since start
           as a float.  Pauses are taken into account.
        """
        return self.adjusted_time() - self.start_time

    def remaining_time(self):
        """Return our best estimate of the remaining duration, or None
        if we have no bases for guessing."""
        if self.end_times is None:
            return 2 * self.elapsed_time()  # We have not started the first module yet
        else:
            module_index = self.current_module.module_num - 1
            index = self.image_set_index * self.num_modules + module_index
            durations = (self.end_times[1:] - self.end_times[:-1]).reshape(
                self.num_image_sets, self.num_modules
            )
            per_module_estimates = numpy.zeros(self.num_modules)
            per_module_estimates[:module_index] = numpy.median(
                durations[: self.image_set_index + 1, :module_index], 0
            )
            current_module_so_far = self.elapsed_time() - self.end_times[1 + index - 1]
            if self.image_set_index > 0:
                per_module_estimates[module_index:] = numpy.median(
                    durations[: self.image_set_index, module_index:], 0
                )
                per_module_estimates[module_index] = max(
                    per_module_estimates[module_index], current_module_so_far
                )
            else:
                # Guess that the modules that haven't finished yet are
                # as slow as the slowest one we've seen so far.
                per_module_estimates[module_index] = current_module_so_far
                per_module_estimates[module_index:] = per_module_estimates[
                    : module_index + 1
                ].max()
            per_module_estimates[:module_index] *= (
                self.num_image_sets - self.image_set_index - 1
            )
            per_module_estimates[module_index:] *= (
                self.num_image_sets - self.image_set_index
            )
            per_module_estimates[module_index] -= current_module_so_far
            return per_module_estimates.sum()

    def remaining_time_multiprocessing(self):
        """Return our best estimate of the remaining duration, or None
        if we have no bases for guessing."""
        if (self.end_times is None) or (self.num_received == 0):
            return 2 * self.elapsed_time()  # We have not started the first module yet
        else:
            expected_per_job = self.end_times[self.num_received] / self.num_received
            return expected_per_job * (self.num_jobs - self.num_received)
