#!/usr/bin/env python
#
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2014 Broad Institute
# All rights reserved.
#
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org
#

from optparse import OptionParser
import os
import subprocess
import sys
import time
import vcloud

def login_to_vcloud():
    with open(os.path.join(os.path.dirname(sys.argv[0]),
                           "vcloud-password.txt")) as f:
        password = f.readline().rstrip()
    session = vcloud.Session("https://vcd.broadinstitute.org/api", "cpbuild", 
                             "Imaging", password)
    return session

def deploy_vm(session, template_name):
    org = session.organization()
    catalog = org.catalogs["Build Catalog"]
    item = catalog[template_name]
    vdc = org.virtual_data_center()
    vapp = vdc.deploy(item)
    return vapp

def get_ip_address(vapp):
    """Wait for and get IP address."""
    while not vapp.ip_address:
        time.sleep(1)
        vapp.reload()
    return vapp.ip_address

def delete_vm(vapp):
    vapp.reload()
    if vapp.deployed:
        vapp.undeploy()
    vapp.delete()

parser = OptionParser("""usage: %prog TEMPLATE [COMMAND...]
Runs COMMAND with the IP address as an additional argument.
If COMMAND is not given, just start the VM and print the IP address.""")
options, args = parser.parse_args()
if len(args) < 1:
    parser.print_usage()
    sys.exit(1)
template_name = args[0]
command = args[1:]

session = login_to_vcloud()
vapp = deploy_vm(session, template_name)
ip_address = get_ip_address(vapp)

if len(command) == 0:
    print ip_address
else:
    return_code = subprocess.call(command + [ip_address])
    delete_vm(vapp)
    sys.exit(return_code)
