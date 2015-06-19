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
-- run_base
--
-- Represents a command to be run on a cluster node
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `run_base` (
  `run_id` int(11) NOT NULL AUTO_INCREMENT,
  `batch_id` int(11) NOT NULL,
  `run_type` varchar(16) NOT NULL,
  `command` text NOT NULL,
  PRIMARY KEY (`run_id`),
  KEY `run_batch_id_idx` (`batch_id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;
  
-------------------------------------------------------------
--
-- run_cellprofiler
--
-- Represents the image sets to be run on CellProfiler
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `run_cellprofiler` (
  `run_id` int(11) NOT NULL,
  `bstart` int(11) NOT NULL,
  `bend` int(11) NOT NULL,
  PRIMARY KEY (`run_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-------------------------------------------------------------
--
-- run_sql
--
-- Represents the SQL file to be run
--
-- The sql_filename does not include the path
-- sql_path = os.path.join(batch.data_dir, sql_filename)
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `run_sql` (
    `run_id` int(11) NOT NULL,
    `sql_filename` varchar(256),
    PRIMARY KEY (`run_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-------------------------------------------------------------
--
-- run
--
-- The legacy view of CellProfiler runs
-------------------------------------------------------------
DROP TABLE IF EXISTS run CASCADE;

CREATE OR REPLACE VIEW `run` AS
SELECT rb.batch_id as batch_id,
       rb.run_id as run_id,
       rc.bstart as bstart,
       rc.bend as bend,
       NULL as bgroup,
       rb.command as command
  FROM run_base rb JOIN run_cellprofiler rc
    ON rb.run_id = rc.run_id
 WHERE rb.run_type = 'CellProfiler';
 
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
  KEY `job_job_id_idx` (`job_id`),
  KEY `job_created_idx` (`created`)
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
  `status` varchar(16) NOT NULL,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY `job_status_id_fk` (`run_id`, `job_id`),
  KEY `job_status_created_k` (`created`)
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8;
