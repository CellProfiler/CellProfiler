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
	   rb.run_type as run_type,
       rb.command as command,
       rc.bstart as bstart,
       rc.bend as bend,
       rs.sql_filename as sql_filename
  FROM run_base rb 
  LEFT JOIN run_cellprofiler rc ON rb.run_id = rc.run_id
  LEFT JOIN run_sql rs ON rs.run_id = rc.run_id;
 
-------------------------------------------------------------
--
-- batch_array
--
-- An array of tasks to be performed using a job array
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `batch_array` (
  -- The batch array ID uniquely identifies the batch array --
  `batch_array_id` int(11) NOT NULL AUTO_INCREMENT,
  -- The batch on whose behalf the batch array is issued --
  `batch_id` int(11) NOT NULL,
  PRIMARY KEY (`batch_array_id`),
  KEY `batch_array_batch_id_fk` (`batch_id`)
)  ENGINE=InnoDB DEFAULT CHARSET=utf8;
-------------------------------------------------------------
--
-- batch_array_task
--
-- A task within a job array tied to a run. A job array
-- can be launched against a batch and the task ids can
-- be used to fetch the command to run via a join to the run_id
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `batch_array_task` (
  `batch_array_task_id` int(11) NOT NULL auto_increment,
  -- The batch array containing the run
  `batch_array_id` int(11) NOT NULL,
  -- The task ID for the task within the job array
  `task_id` int(11) NOT NULL,
  -- The run ID associated with task # task_id in the batch
  `run_id`  int(11) NOT NULL,
  PRIMARY KEY (`batch_array_task_id`),
  KEY `batch_array_task_batch_array_fk` (`batch_array_id`, `task_id`),
  KEY `batch_array_task_run_id_fk` (`run_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-------------------------------------------------------------
--
-- job
--
-- A job run on a cluster node
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `job` (
  -- the database identifier for the record
  `job_record_id` int(11) NOT NULL AUTO_INCREMENT,
  -- the job ID as reported by qsub or from the JOB_ID environment variable
  `job_id` int(11) NOT NULL,
  -- the batch array for the job --
  `batch_array_id` int(11) NOT NULL,
  -- timestamp of job's creation
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`job_record_id`),
  KEY `job_batch_array_id_fk` (`batch_array_id`),
  KEY `job_job_id_idx` (`job_id`),
  KEY `job_created_idx` (`created`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-------------------------------------------------------------
--
-- job_task
--
-- A task within an array job
--
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `job_task` (
  -- The primary key for a job task --
  `job_task_id` int(11) NOT NULL AUTO_INCREMENT,
  -- The link to the database record for the job --
  `job_record_id` int(11) NOT NULL,
  -- The task ID within the job --
  `batch_array_task_id` int(11) NOT NULL,
  PRIMARY KEY (`job_task_id`),
  KEY `job_task_job_fk` (`job_record_id`),
  KEY `job_task_batch_array_task_fk` (`batch_array_task_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-------------------------------------------------------------
--
-- task_status
--
-- The status of a job's task
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

CREATE TABLE IF NOT EXISTS `task_status` (
  `task_status_id` int(11) NOT NULL AUTO_INCREMENT,
  `job_task_id` int(11) NOT NULL,
  `status` varchar(16) NOT NULL,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`task_status_id`),
  KEY `task_status_job_task_fk` (`job_task_id`, `created`)
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8;
  
-------------------------------------------------------------
--
-- run_job_status
--
-- This view shows the last job and its status for every
-- run.
--
-------------------------------------------------------------

CREATE OR REPLACE VIEW run_job_status AS
SELECT rb.batch_id as batch_id, rb.run_id as run_id, rb.run_type,
       rb.command,
       rs.sql_filename as sql_filename, 
       rc.bstart as bstart, rc.bend as bend,
       bat.batch_array_task_id as batch_array_task_id,
       bat.batch_array_id as batch_array_id, bat.task_id as task_id,
       j.job_record_id as job_record_id, j.job_id as job_id,
       j.created as job_created,
       jt.job_task_id as job_task_id, 
       js.task_status_id as task_status_id, 
       js.status as status, js.created as status_updated
  FROM run_base rb
  LEFT JOIN run_cellprofiler rc on rb.run_id = rc.run_id
  LEFT JOIN run_sql rs on rb.run_id = rs.run_id
  JOIN batch_array_task bat on bat.run_id = rb.run_id
  JOIN job j on bat.batch_array_id = j.batch_array_id
  JOIN job_task jt 
    ON jt.job_record_id = j.job_record_id 
   AND jt.batch_array_task_id = bat.batch_array_task_id
  JOIN task_status js ON jt.job_task_id = js.job_task_id
 WHERE NOT EXISTS 
 (SELECT 'x' 
    FROM task_status js2
    JOIN job_task jt2 on js2.job_task_id = jt2.job_task_id
    JOIN job j2 on j2.job_record_id = jt2.job_record_id
    JOIN batch_array_task bat2
      ON bat2.batch_array_id = j2.batch_array_id
     AND bat2.batch_array_task_id = jt2.batch_array_task_id
  WHERE bat2.run_id = rb.run_id AND js2.created > js.created);

-------------------------------------------------------------
--
-- task_host
--
-- This table keeps track of the host for a task and any
-- resources (like XVFB server numbers) that might need
-- to be reserved
--
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `task_host` (
  `job_task_id` int(11) NOT NULL,
  `hostname` varchar(64) NOT NULL,
  `xvfb_server` int(11),
  KEY `job_host_id_fk` (`job_task_id`),
  KEY `job_host_hostname_k` (`hostname`)
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8;
