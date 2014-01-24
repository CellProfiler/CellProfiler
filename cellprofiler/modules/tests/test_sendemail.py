'''test_sendemail - test the SendEmail module

'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2014 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

import asyncore
from cStringIO import StringIO
import Queue
import smtpd
import threading
import unittest

import cellprofiler.pipeline as cpp
import cellprofiler.measurements as cpmeas
import cellprofiler.workspace as cpw
import cellprofiler.objects as cpo
import cellprofiler.modules.sendemail as SE

SENDER = "sender@cellprofiler.org"
def recipient_addr(idx):
    return "recipient%d@cellprofiler.org" % (idx+1)

class MockSMTPServer(smtpd.SMTPServer):
    def __init__(self, queue):
        smtpd.SMTPServer.__init__(self, ('localhost', 0), None)
        self.queue = queue
        
    def process_message(self, peer, mailfrom, rcpttos, data):
        self.queue.put((peer, mailfrom, rcpttos, data))
            
class TestSendEmail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queue = Queue.Queue()
        cls.server = MockSMTPServer(cls.queue)
        cls.port = cls.server.socket.getsockname()[1]
        cls.thread = threading.Thread(target = cls.threadFn)
        cls.thread.setDaemon(True)
        cls.thread.start()
        
    @classmethod
    def threadFn(cls):
        asyncore.loop(map = cls.server._map)
        
    @classmethod
    def tearDownClass(cls):
        cls.server.close()
        cls.thread.join()
        
    def recv(self, module, whens, 
             expected_subject=None, 
             expected_body = None):
        '''Receive an email from the daemon and validate'''
        try:
            peer, mailfrom, rcpttos, data = self.queue.get(timeout=10)
        except Queue.Empty:
            self.fail("Mail daemon timeout")
        self.assertEqual(mailfrom, module.from_address)
        self.assertEqual(len(rcpttos), len(module.recipients))
        self.assertSetEqual(
            set(rcpttos), 
            set(map(lambda x: x.recipient.value, module.recipients)))
        lines = data.split("\n")
        sep = lines.index("")
        header = lines[:sep]
        body = "\n".join(lines[(sep+1):])
        if expected_body is None:
            expected_body = "\n".join([when.message.value for when in whens])
        self.assertEqual(body, expected_body)
        subject_headers = [line.split(': ', 1)[1] for line in lines
                           if line.startswith("Subject: ")]
        self.assertEqual(len(subject_headers), 1)
        if expected_subject is None:
            expected_subject = module.subject.value
        self.assertEqual(subject_headers[0], expected_subject)
    
    def poll(self):
        try:
            peer, mailfrom, rcpttos, data = self.queue.get(timeout=1)
        except Queue.Empty:
            return
        self.fail("Received unexpected email")
        
    def make_workspace(self, image_numbers, 
                       group_numbers = None, 
                       group_indexes = None,
                       n_recipients = 1):
        m = cpmeas.Measurements()
        if group_numbers is None:
            group_numbers = [1] * len(image_numbers)
            group_indexes = range(1, len(image_numbers)+1)
        for image_number, group_number, group_index in zip(
            image_numbers, group_numbers, group_indexes):
            m[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_number] = group_number
            m[cpmeas.IMAGE, cpmeas.GROUP_INDEX, image_number] = group_index
            
        module = SE.SendEmail()
        module.module_num = 1
        module.from_address.value = SENDER
        module.smtp_server.value = 'localhost'
        module.port.value = self.port
        module.connection_security.value = SE.C_NONE
        while len(module.recipients) < n_recipients:
            module.add_recipient()
        for i, recipient in enumerate(module.recipients):
            recipient.recipient.value = recipient_addr(i)
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, m, cpo.ObjectSet(),
                                  m, None)
        return workspace, module
        
        
    def test_01_02_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20130220214757
ModuleCount:5
HasImagePlaneDetails:False

SendEmail:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Hidden:2
    Hidden:6
    Sender address:sender@cellprofiler.org
    Subject line:Hello, world
    Server name:smtp.cellprofiler.org
    Port:587
    Select connection security:STARTTLS
    Username and password required to login?:Yes
    Username:sender@cellprofiler.org
    Password:rumplestiltskin
    Recipient address:recipient@broadinstitute.org
    Recipient address:recipient@cellprofiler.org
    When should the email be sent?:After first cycle
    Image cycle number:1
    Image cycle count:1
    Message text:First cycle
    When should the email be sent?:After last cycle
    Image cycle number:1
    Image cycle count:1
    Message text:Last cycle
    When should the email be sent?:After group start
    Image cycle number:1
    Image cycle count:1
    Message text:Group start
    When should the email be sent?:After group end
    Image cycle number:1
    Image cycle count:1
    Message text:Group end
    When should the email be sent?:Every # of cycles
    Image cycle number:1
    Image cycle count:5
    Message text:Every fifth cycle
    When should the email be sent?:After cycle #
    Image cycle number:17
    Image cycle count:1
    Message text:Cycle 17
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        assert isinstance(module, SE.SendEmail)
        self.assertEqual(module.from_address, "sender@cellprofiler.org")
        self.assertEqual(module.subject, "Hello, world")
        self.assertEqual(module.smtp_server, "smtp.cellprofiler.org")
        self.assertEqual(module.port, 587)
        self.assertEqual(module.connection_security, SE.C_STARTTLS)
        self.assertTrue(module.use_authentication)
        self.assertEqual(module.username, "sender@cellprofiler.org")
        self.assertEqual(module.password, "rumplestiltskin")
        self.assertEqual(len(module.recipients), 2)
        self.assertEqual(module.recipients[0].recipient, "recipient@broadinstitute.org")
        self.assertEqual(module.recipients[1].recipient, "recipient@cellprofiler.org")
        self.assertEqual(len(module.when), 6)
        self.assertEqual(module.when[0].choice, SE.S_FIRST)
        self.assertEqual(module.when[0].message, "First cycle")
        self.assertEqual(module.when[1].choice, SE.S_LAST)
        self.assertEqual(module.when[2].choice, SE.S_GROUP_START)
        self.assertEqual(module.when[3].choice, SE.S_GROUP_END)
        self.assertEqual(module.when[4].choice, SE.S_EVERY_N)
        self.assertEqual(module.when[4].image_set_count, 5)
        self.assertEqual(module.when[5].choice, SE.S_CYCLE_N)
        self.assertEqual(module.when[5].image_set_number, 17)
        
    def test_02_01_first(self):
        workspace, module = self.make_workspace([1, 2])
        
        assert isinstance(module, SE.SendEmail)
        module.when[0].choice.value = SE.S_FIRST
        module.when[0].message.value = "First cycle"
        workspace.measurements.next_image_set(1)
        module.run(workspace)
        self.recv(module, [module.when[0]])
        workspace.measurements.next_image_set(2)
        module.run(workspace)
        self.poll()
        
    def test_02_02_last(self):
        workspace, module = self.make_workspace([1, 2])
        
        assert isinstance(module, SE.SendEmail)
        module.when[0].choice.value = SE.S_LAST
        module.when[0].message.value = "Last cycle"
        workspace.measurements.next_image_set(1)
        module.run(workspace)
        self.poll()
        workspace.measurements.next_image_set(2)
        module.run(workspace)
        self.poll()
        module.post_run(workspace)
        self.recv(module, [module.when[0]])

    def test_02_03_cycle(self):
        workspace, module = self.make_workspace([1, 2, 3])
        
        assert isinstance(module, SE.SendEmail)
        module.when[0].choice.value = SE.S_CYCLE_N
        module.when[0].image_set_number.value = 2
        module.when[0].message.value = "Cycle 2"
        workspace.measurements.next_image_set(1)
        module.run(workspace)
        self.poll()
        workspace.measurements.next_image_set(2)
        module.run(workspace)
        self.recv(module, [module.when[0]])
        workspace.measurements.next_image_set(3)
        module.run(workspace)
        self.poll()
        
    def test_02_04_every_n(self):
        image_numbers = [1, 2, 3, 4, 5, 6]
        workspace, module = self.make_workspace(image_numbers)
        
        assert isinstance(module, SE.SendEmail)
        module.when[0].choice.value = SE.S_EVERY_N
        module.when[0].image_set_count.value = 3
        module.when[0].message.value = "Every 3"
        for i in image_numbers:
            workspace.measurements.next_image_set(i)
            module.run(workspace)
            if i % 3 == 0:
                self.recv(module, [module.when[0]])
            else:
                self.poll()

    def test_02_05_first_group(self):
        image_numbers = [1, 2, 3, 4, 5, 6]
        group_numbers = [1, 1, 2, 2, 2, 2]
        group_indexes = [1, 2, 1, 2, 3, 4]
        workspace, module = self.make_workspace(image_numbers, 
                                                group_numbers,
                                                group_indexes)
        
        assert isinstance(module, SE.SendEmail)
        module.when[0].choice.value = SE.S_GROUP_START
        module.when[0].message.value = "First in group"
        for i, gi in zip(image_numbers, group_indexes):
            workspace.measurements.next_image_set(i)
            module.run(workspace)
            if gi == 1:
                self.recv(module, [module.when[0]])
            else:
                self.poll()
        
    def test_02_06_last_group(self):
        image_numbers = [1, 2, 3, 4, 5, 6]
        group_numbers = [1, 1, 2, 2, 2, 2]
        group_indexes = [1, 2, 1, 2, 3, 4]
        last_in_group = [0, 1, 0, 0, 0, 1]
        workspace, module = self.make_workspace(image_numbers, 
                                                group_numbers,
                                                group_indexes)
        
        assert isinstance(module, SE.SendEmail)
        module.when[0].choice.value = SE.S_GROUP_END
        module.when[0].message.value = "Last in group"
        for i, group_index, flag in zip(
            image_numbers, group_indexes, last_in_group):
            workspace.measurements.next_image_set(i)
            if group_index == 1:
                module.prepare_group(
                    workspace, None, 
                    [1, 2] if i == 1 else [3, 4, 5, 6])
            module.run(workspace)
            if flag == 1:
                self.recv(module, [module.when[0]])
            else:
                self.poll()
                
    def test_03_01_two_hits(self):
        workspace, module = self.make_workspace([1, 2])
        assert isinstance(module, SE.SendEmail)
        module.when[0].choice.value = SE.S_GROUP_START
        module.when[0].message.value = "Group start"
        module.add_when()
        module.when[1].choice.value = SE.S_CYCLE_N
        module.when[1].image_set_number.value = 100
        module.when[1].message.value = "Don't send me"
        module.add_when()
        module.when[2].choice.value = SE.S_FIRST
        module.when[2].message.value = "First image"
        workspace.measurements.next_image_set(1)
        module.run(workspace)
        self.recv(module, [module.when[0], module.when[2]])
                                                
    def test_03_02_metadata_in_subject(self):
        workspace, module = self.make_workspace([1, 2])
        assert isinstance(module, SE.SendEmail)
        module.when[0].choice.value = SE.S_EVERY_N
        module.when[0].image_set_count.value = 1
        module.when[0].message.value = "First image"
        module.subject.value = r"Well \g<Well>"
        m = workspace.measurements
        m[cpmeas.IMAGE, "Metadata_Well", 1] = "A01"
        m[cpmeas.IMAGE, "Metadata_Well", 2] = "A02"
        m.next_image_set(1)
        module.run(workspace)
        self.recv(module, [module.when[0]],
                  expected_subject = "Well A01")
        m.next_image_set(2)
        module.run(workspace)
        self.recv(module, [module.when[0]],
                  expected_subject = "Well A02")
        
    def test_03_03_metadata_in_message(self):
        workspace, module = self.make_workspace([1, 2])
        assert isinstance(module, SE.SendEmail)
        module.when[0].choice.value = SE.S_EVERY_N
        module.when[0].image_set_count.value = 1
        module.when[0].message.value = r"Well \g<Well>"
        m = workspace.measurements
        m[cpmeas.IMAGE, "Metadata_Well", 1] = "A01"
        m[cpmeas.IMAGE, "Metadata_Well", 2] = "A02"
        m.next_image_set(1)
        module.run(workspace)
        self.recv(module, [module.when[0]],
                  expected_body = "Well A01")
        m.next_image_set(2)
        module.run(workspace)
        self.recv(module, [module.when[0]],
                  expected_body = "Well A02")
        
