#!/usr/bin/env python

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

parser = OptionParser("""usage: %prog TEMPLATE COMMAND...
Runs COMMAND with the IP address as an additional argument.""")
options, args = parser.parse_args()
if len(args) < 2:
    parser.print_usage()
    sys.exit(1)
template_name = args[0]
command = args[1:]

session = login_to_vcloud()
vapp = deploy_vm(session, template_name)
ip_address = get_ip_address(vapp)

return_code = subprocess.call(command + [ip_address])

delete_vm(vapp)

sys.exit(return_code)
