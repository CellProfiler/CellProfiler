-- CellProfiler is distributed under the GNU General Public License.
-- See the accompanying file LICENSE for details.
--
-- Copyright (c) 2003-2009 Massachusetts Institute of Technology
-- Copyright (c) 2009-2015 Broad Institute
-- All rights reserved.
--
-- Please see the AUTHORS file for credits.
--
-- Website: http://www.cellprofiler.org


-------------------------------------------------------------
--
-- batch
--
-- Each row of the batch table corresponds to an analysis
-- driven by a Batch_data.h5 file.
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `batch` (
  -- The identifier of the batch
  `batch_id` int(11) NOT NULL AUTO_INCREMENT,
  -- Report status to this email address
  `email` text,
  -- The location of the Batch_data.h5 file and the job log files
  `data_dir` text,
  -- Run the batch on this queue
  `queue` text,
  -- # of image sets / run
  `batch_size` int(11) DEFAULT NULL,
  -- 1 to write the measurements .h5 file, 0 to exclude it
  `write_data` tinyint(1) DEFAULT NULL,
  -- (obsolete)
  `timeout` float DEFAULT NULL,
  -- The root directory of the CellProfiler version
  `cpcluster` text,
  -- The # of megabytes of memory to reserve for the VM
  `memory_limit` float NOT NULL DEFAULT '2000',
  -- Charge the job to this group
  `project` varchar(256) NOT NULL DEFAULT 'imaging',
  -- The priority of the job
  `priority` int(4) DEFAULT '50',
  -- timestamp of batch's creation
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`batch_id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;

-------------------------------------------------------------
--
-- run
--
-- Each row of the run table represents a group of image sets
-- to be executed by a job
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `run` (
  `run_id` int(11) NOT NULL AUTO_INCREMENT,
  `batch_id` int(11) NOT NULL,
  `bstart` int(11) NOT NULL,
  `bend` int(11) NOT NULL,
  `bgroup` text,
  PRIMARY KEY (`run_id`),
  KEY `run_batch_id_idx` (`batch_id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;

-------------------------------------------------------------
--
-- job
--
-- A job run on a cluster node
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `job` (
  -- the job ID as reported by qsub or from the JOB_ID environment variable
  `job_id` int(11) NOT NULL,
  -- the associated run ID
  `run_id` int(11) NOT NULL,
  -- timestamp of job's creation
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY `job_run_id_fk` (`run_id`),
  KEY `job_job_id_idx` (`job_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-------------------------------------------------------------
--
-- job_status
--
-- The status of the job
--
-- SUBMITTED - the job has been submitted to the cluster
-- RUNNING - the script has started running
-- ERROR - CellProfiler exited with an error status
-- DONE - CellProfiler exited cleanly
-- ABORTED - qdel or similar was run in an attempt to kill the job
--
-- typical usage might be:
-- select status from job_status where job_id = # order by created desc limit 1
--
-- to get the last reported status
-------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `job_status` (
  -- the job ID of the associated job
  `job_id` int(11) NOT NULL,
  `run_id` int(11) NOT NULL,
  `status` text NOT NULL,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY `job_status_id_fk` (`run_id`, `job_id`)
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8;
