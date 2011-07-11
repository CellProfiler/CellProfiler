'''<b>SendEmail</b> send emails to a specified address at desired stages
of processing
<hr>

This module sends email about the current
progress of the image processing. You can specify how often emails
are sent out (for example, after the first cycle, after the last cycle,
after every N cycles, after N cycles). This module should be placed at
the point in the pipeline when you want the emails to be sent. If email
sending fails for any reason, a warning message will appear but
processing will continue regardless.'''

__version__ = "$Revision$"

import logging
import email.message
import os
import traceback
import smtplib
import sys

logger = logging.getLogger(__name__)
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
from cellprofiler.gui.help import USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF

S_FIRST = "After first cycle"
S_LAST = "After last cycle"
S_GROUP_START = "After group start"
S_GROUP_END = "After group end"
S_EVERY_N = "Every # of cycles"
S_CYCLE_N = "After cycle #"
S_ALL = [S_FIRST, S_LAST, S_GROUP_START, S_GROUP_END, S_EVERY_N, S_CYCLE_N]

'''If you make changes to the wording above, please enter the translation below

Every old and new text string should appear as a key in this dictionary
with one of the symbolic values listed above.
'''
S_DICTIONARY = {
    "After first cycle": S_FIRST,
    "After last cycle": S_LAST,
    "After group start": S_GROUP_START,
    "After group end": S_GROUP_END,
    "Every # of cycles": S_EVERY_N,
    "After cycle #": S_CYCLE_N
    }

K_EMAIL = "Email"
K_CYCLE = "Cycle"
K_RESULT = "Result"

'''Number of settings in an event'''
EVENT_SETTING_COUNT = 4

class SendEmail(cpm.CPModule):
    
    module_name = "SendEmail"
    category = "Other"
    variable_revision_number = 1
    
    def create_settings(self):
        '''Create the UI settings for this module'''
        
        self.recipients = []
        self.recipient_count = cps.HiddenCount(self.recipients)
        self.add_recipient(False)
        self.add_recipient_button = cps.DoSomething(
            "Add a recipient address.",
            "Add address",
            self.add_recipient)
        
        if sys.platform.startswith("win"):
            user = os.environ.get("USERNAME","yourname@yourdomain")
        else:
            user = os.environ.get("USER","yourname@yourdomain")
            
        self.from_address = cps.Text(
            "Sender address", user,
            doc="""Enter the address for the email's "From" field.""")
        
        self.subject = cps.Text(
            "Subject line","CellProfiler notification",
            metadata=True,
            doc="""Enter the text for the email's subject line. If you have metadata 
            associated with your images, you can use metadata tags here. %(USING_METADATA_TAGS_REF)s<br>
            For instance, if you have plate metadata,
            you might use the line, "CellProfiler: processing plate " and insert the metadata tag
            for the plate at the end. %(USING_METADATA_HELP_REF)s."""%globals())
        
        self.smtp_server = cps.Text(
            "SMTP server", "mail",
            doc="""Enter the address of your SMTP server. You can ask your
            network administrator for your outgoing mail server which is often
            made up of part of your email address, e.g., 
            "Something@university.org". You might be able to find this information
            by checking your settings or preferences in whatever email program
            you use.""")
        
        self.port = cps.Integer(
            "SMTP port", smtplib.SMTP_PORT, 0, 65535,
            doc="""Enter your server's SMTP port. The default (25) is the
            port used by most SMTP servers. Your network administrator may
            have set up SMTP to use a different port.""")
        self.when = []
        self.when_count = cps.HiddenCount(self.when)
        self.add_when(False)
        
        self.add_when_button = cps.DoSomething(
            "Add another email event","Add event", self.add_when,
            doc="""Press this button to add another event or condition.
            <b>SendEmail</b> will send an email when this event happens""")
        
    def add_recipient(self, can_delete = True):
        '''Add a recipient for the email to the list of emails'''
        group = cps.SettingsGroup()
        group.append("recipient", cps.Text(
            "Recipient address","recipient@domain",
            doc="""Enter the address to which the messages will be sent."""))
        if can_delete:
            group.append("remover", cps.RemoveSettingButton(
                "Remove above recipient", "Remove recipient", 
                self.recipients, group,
                doc="""Press this button to remove the above recipient from
                the list of people to receive the email"""))
        self.recipients.append(group)
            
    def add_when(self, can_delete = True):
        group = cps.SettingsGroup()
        group.append("choice", cps.Choice(
            "When should the email be sent?", S_ALL,
            doc="""Select the kind of event that causes
            <b>SendEmail</b> to send an email. You have the following choices:
            <br><ul><li><i>After first cycle:</i> Send an email during
            processing of the first image cycle.</li>
            <li><i>After last cycle:</i> Send an email after all processing
            is complete.</li>
            <li><i>After group start:</i> Send an email during the first
            cycle of each group of images.</li>
            <li><i>After group end:</i> Send an email after all processing
            for a group is complete.</li>
            <li><i>Every # of cycles</i> Send an email each time a certain
            number of image cycles have been processed. You will be prompted
            for the number of image cycles if you select this choice.</li>
            <li><i>After cycle #:</i> Send an email after the given number
            of image cycles have been processed. You will be prompted for
            the image cycle number if you select this choice. You can add
            more events if you want emails after more than one image cycle.</li>
            </ul>"""))
        
        group.append("image_set_number", cps.Integer(
            "Image cycle number", 1, minval = 1,
            doc='''
            <i>(Used only if sending email after a particular cycle number)</i><br>
            Send an email during processing of the given image cycle.
            For instance, if you enter 4, then <b>SendEmail</b>
            will send an email during processing of the fourth image cycle.'''))
        
        group.append("image_set_count", cps.Integer(
            "Image cycle count", 1, minval = 1,
            doc='''
            <i>(Used only if sending email after every N cycles)</i><br>
            Send an email each time this number of image cycles have
            been processed. For instance, if you enter 4,
            then <b>SendEmail</b> will send an email during processing of
            the fourth, eighth, twelfth, etc. image cycles.'''))
        
        group.append("message", cps.Text(
            "Message text","Notification from CellProfiler",
            metadata=True,
            doc="""The body of the message sent from CellProfiler.
            Your message can include metadata values. For instance,
            if you group by plate and want to send an email after processing each
            plate, you could use the message  
            "Finished processing plate \\g<Plate>". """))
        
        if can_delete:
            group.append("remover", cps.RemoveSettingButton(
                "Remove this email event", "Remove event", self.when, group))
        group.append("divider", cps.Divider())
        self.when.append(group)
        
    def settings(self):
        '''The settings as saved in the pipeline'''
        result = [ self.recipient_count, self.when_count, 
                   self.from_address, self.subject, self.smtp_server,
                   self.port]
        for group in self.recipients + self.when:
            result += group.pipeline_settings()
        return result
    
    def visible_settings(self):
        '''The settings as displayed in the UI'''
        result = []
        for group in self.recipients:
            result += group.visible_settings()
        result += [self.add_recipient_button, self.from_address, 
                   self.subject, self.smtp_server, self.port]
        for group in self.when:
            result += [ group.choice ]
            if group.choice == S_CYCLE_N:
                result += [ group.image_set_number ]
            elif group.choice == S_EVERY_N:
                result += [ group.image_set_count ]
            result += [group.message]
            if hasattr(group, "remover"):
                result += [group.remover]
            result += [group.divider]
        result += [self.add_when_button]
        return result
    
    def is_interactive(self):
        return False
    
    def prepare_run(self, workspace):
        '''Prepare to run the first image in the image set'''
        email_me = [ group.message.value for group in self.when
                     if group.choice == S_FIRST ]
        
        d = self.get_dictionary(workspace.image_set_list)
        d[K_EMAIL] = email_me
        d[K_CYCLE] = 0
        d[K_RESULT] = "No email sent"
        return True
    
    def prepare_group(self, pipeline, image_set_list, grouping, image_numbers):
        '''Prepare to run the first image in the group'''
        email_me = [group.message.value for group in self.when
                    if group.choice == S_GROUP_START ]
        if email_me:
            self.get_dictionary(image_set_list)[K_EMAIL] += email_me
            
    def run(self, workspace):
        '''Run every image set'''
        #
        # This is the module's dictionary for the run
        #
        d = self.get_dictionary(workspace.image_set_list)
        email_me = d[K_EMAIL]
        d[K_EMAIL] = []
        cycle = d[K_CYCLE]
        d[K_CYCLE] += 1
        
        for group in self.when:
            if (group.choice == S_CYCLE_N and 
                cycle == group.image_set_number.value - 1):
                email_me.append(group.message.value)
            if (group.choice == S_EVERY_N and
                cycle % group.image_set_count.value == 
                group.image_set_count.value - 1):
                email_me.append(group.message.value)
        if len(email_me) > 0:
            d[K_RESULT] = self.send_email(workspace, email_me)
        if workspace.frame is not None:
            workspace.display_data.result = d[K_RESULT]
            
    def display(self, workspace):
        import matplotlib
        message = workspace.display_data.result
        figure = workspace.create_or_find_figure(title="SendEmail, image cycle #%d"%(
                workspace.measurements.image_set_number))
        
        mfigure = figure.figure
        assert isinstance(mfigure, matplotlib.figure.Figure)
        mfigure.clf()
        mfigure.text(.1, .1, workspace.display_data.result)
    
    def post_group(self, workspace, grouping):
        '''Possibly send an email after finishing a group'''
        email_me = [group.message.value for group in self.when
                    if group.choice == S_GROUP_END ]
        if len(email_me) > 0:
            d = self.get_dictionary(workspace.image_set_list)
            d[K_RESULT] = self.send_email(workspace, email_me)
            
    def post_run(self, workspace):
        '''Possibly send an email as we finish the run'''
        email_me = [group.message.value for group in self.when
                    if group.choice == S_LAST ]
        if len(email_me) > 0:
            d = self.get_dictionary(workspace.image_set_list)
            d[K_RESULT] = self.send_email(workspace, email_me)

    def send_email(self, workspace, messages):
        '''Send an email according to the settings'''
        
        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)
        
        message = email.message.Message()
        who_from = self.from_address.value
        who_to = [group.recipient.value for group in self.recipients]
        subject = measurements.apply_metadata(self.subject.value)
        
        message["Content-Type"] = "text/plain"
        message["From"] = who_from
        message["To"] = ",".join(who_to)
        message["Subject"] = subject
        
        payload = []
        for msg in messages:
            msg = measurements.apply_metadata(msg)
            payload.append(msg)
        message.set_payload("\n".join(payload))
        
        server = smtplib.SMTP(self.smtp_server.value, self.port.value)
        try:
            server.sendmail(who_from, who_to, message.as_string())
            return message.as_string()
        except:
            logger.error("Failed to send mail", exc_info=True)
            return "Failed to send mail"
    
    def prepare_settings(self, setting_values):
        '''Adjust the numbers of recipients and events according to the settings'''
        
        nrecipients = int(setting_values[0])
        nevents = int(setting_values[1])
        
        if len(self.recipients) > nrecipients:
            del self.recipients[nrecipients:]
        while len(self.recipients) < nrecipients:
            self.add_recipient()
            
        if len(self.when) > nevents:
            del self.when[nevents:]
        while len(self.when) < nevents:
            self.add_when()
            
    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        '''Upgrade the pipeline settings from a previous CP version
        
        setting_values - one string per setting
        variable_revision_number - revision # of module that did saving
        module_name - name of module that did saving
        from_matlab - true if CP 1.0
        '''
        if from_matlab and variable_revision_number == 1:
            recipients, sender, server, send_first, send_last, \
            image_set_count = setting_values[:6]
            image_set_number = [x for x in setting_values[6:] if x != "0"]
            
            events = []
            if send_first == cps.YES:
                events.append((S_FIRST, "1", "1", "Starting processing"))
            if send_last == cps.YES:
                events.append((S_LAST, "1", "1", "Finished processing"))
            if image_set_count != "0":
                events.append((S_EVERY_N, "1", image_set_count, 
                               "Processed %s images" % image_set_count))
            for n in image_set_number:
                events.append((S_CYCLE_N, n, "0",
                               "Processed cycle %s" % n))
            
            recipients = recipients.split(',')
            setting_values = [ str(len(recipients)),
                               str(len(events)),
                               sender,
                               "CellProfiler notification",
                               server,
                               "25"] + recipients
            for event in events:
                setting_values += event
            variable_revision_number = 1
            from_matlab = False

        #
        # Standardize the event names
        #
        setting_values = list(setting_values)
        event_count = int(setting_values[1])
        event_idx = len(setting_values) - EVENT_SETTING_COUNT * event_count
        for i in range(event_idx,len(setting_values), EVENT_SETTING_COUNT):
            if S_DICTIONARY.has_key(setting_values[i]):
                setting_values[i] = S_DICTIONARY[setting_values[i]]
            
        return setting_values, variable_revision_number, from_matlab
                               
                
