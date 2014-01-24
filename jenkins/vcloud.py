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

import base64
import getpass
from optparse import OptionParser
import os.path
import progressbar
import random
import string
import sys
import time
import xpath
import requests
from xml.dom import minidom

VCLOUD_API_VERSION = "5.1"
VERIFY_SSL = False

def get_text(nodelist):
    """
    Given a list of XML elements, concatenate the ones that are
    text nodes and return as a string.

    >>> doc = minidom.parseString('<description>Foo</description>')
    >>> el = doc.documentElement
    >>> get_text(el.childNodes)
    u'Foo'
    
    """
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)


class Version(object):
    def __init__(self, base_url):
        r = requests.get('%s/versions' % base_url, verify=VERIFY_SSL)
        r.raise_for_status()
        self.doc = minidom.parseString(r.text)

    @property
    def login_url(self):
        vi = xpath.find('//SupportedVersions/VersionInfo[Version=%s]' % VCLOUD_API_VERSION, 
                        self.doc)[0]
        lu = xpath.find('LoginUrl', vi)[0]
        return get_text(lu.childNodes)


class Session(object):
    def __init__(self, base_url, username, organization, password):
        base64string = base64.encodestring('%s@%s:%s' % (username, organization, password)).replace('\n', '')
        headers = {'Accept': 'application/*+xml;version=%s' % VCLOUD_API_VERSION,
                   'Authorization': 'Basic %s' % base64string}
        version = Version(base_url)
        r = requests.post(version.login_url, headers=headers, verify=VERIFY_SSL)
        r.raise_for_status()
        self.x_vcloud_authorization = r.headers['x-vcloud-authorization']
        self.doc = minidom.parseString(r.text)

    def _headers(self, accept_type=None):
        return {'Accept': '%s;version=%s' % ((accept_type or 'application/*+xml'),
                                             VCLOUD_API_VERSION),
                'x-vcloud-authorization': self.x_vcloud_authorization}

    def get(self, url, accept_type=None):
        r = requests.get(url, headers=self._headers(accept_type), verify=VERIFY_SSL)
        r.raise_for_status()
        return minidom.parseString(r.text)

    def post(self, url, body, body_type=None, accept_type=None):
        headers = self._headers(accept_type)
        if body_type:
            headers['Content-Type'] = body_type
        r = requests.post(url, data=body, headers=headers, verify=VERIFY_SSL)
        r.raise_for_status()
        return minidom.parseString(r.text)

    def delete(self, url, accept_type=None):
        r = requests.delete(url, headers=self._headers(accept_type), 
                            verify=VERIFY_SSL)
        r.raise_for_status()
        return minidom.parseString(r.text)

    def organization(self):
        el = xpath.find('//Session/Link[@type="%s"]' % Organization.content_type, self.doc)[0]
        return Organization(self, el.getAttribute('href'))

    def organizations(self):
        url = xpath.find('//Session/Link[@type="application/vnd.vmware.vcloud.orgList+xml"]', session.doc)[0].getAttribute('href')
        doc = self.get(url)
        return dict([(org.getAttribute('name'),
                      Organization(self, org.getAttribute('href')))
                     for org in xpath.find('//OrgList/Org', doc)])


class APIType(object):
    def __init__(self, session, url=None, doc=None, content_type=None):
        assert isinstance(session, Session)
        assert url or doc
        self.session = session
        if doc:
            self.doc = doc
            root = self.doc.documentElement
            self.url = url or root.getAttribute('href')
            self.content_type = content_type or root.getAttribute('type')
        else:
            self.url = url
            self.doc = None

    def reload(self):
        doc = self.session.get(self.url)
        self.doc = doc

    def ensure_loaded(self):
        if self.doc:
            return
        self.reload()

    def toxml(self):
        self.ensure_loaded()
        return self.doc.toxml()

    def _get_text_children(self, xq):
        self.ensure_loaded()
        els = xpath.find(xq, self.doc)
        if els:
            return get_text(els[0].childNodes)
        else:
            return None
    
    def _get_attribute(self, xq, attr, default=None):
        self.ensure_loaded()
        els = xpath.find(xq, self.doc)
        if els:
            return els[0].getAttribute(attr)
        else:
            return default


class Organization(APIType):
    content_type = 'application/vnd.vmware.vcloud.org+xml'
    
    @property
    def catalogs(self):
        self.ensure_loaded()
        els = xpath.find('//Org/Link[@type="%s"]' % Catalog.content_type,
                         self.doc)
        return dict([(el.getAttribute('name'),
                      Catalog(self.session, el.getAttribute('href')))
                     for el in els])

    def virtual_data_center(self):
        vdc_url = self._get_attribute('//Org/Link[@type="%s"]' % VirtualDataCenter.content_type, 'href')
        return VirtualDataCenter(self.session, vdc_url)


class Catalog(APIType):
    content_type = 'application/vnd.vmware.vcloud.catalog+xml'

    @property
    def name(self):
        return self._get_attribute('//Catalog', 'name')

    @property
    def description(self):
        return self._get_text_children('//Catalog/Description')

    _items = None

    def _ensure_items_parsed(self):
        self.ensure_loaded()
        if self._items is None:
            self._items = dict([(item.getAttribute('name'), 
                                 item.getAttribute('href'))
                           for item in xpath.find('//Catalog/CatalogItems/CatalogItem', 
                                                  self.doc)])

    def __getitem__(self, name):
        self._ensure_items_parsed()
        return CatalogItem(self.session, self._items[name])

    def keys(self):
        self._ensure_items_parsed()
        return self._items.keys()


class CatalogItem(APIType):
    content_type = 'application/vnd.vmware.vcloud.catalogItem+xml'

    @property
    def name(self):
        self.ensure_loaded()
        return self._get_attribute('//CatalogItem', 'name')

    @property
    def template_url(self):
        self.ensure_loaded()
        return self._get_attribute('//CatalogItem/Entity', 'href')

    def template(self):
        return VAppTemplate(self.session, self.template_url)


class VAppTemplate(APIType):
    content_type = 'application/vnd.vmware.vcloud.vAppTemplate+xml'

    @property
    def network_name(self):
        self.ensure_loaded()
        el = xpath.find('//VAppTemplate/Children/Vm/NetworkConnectionSection/NetworkConnection', self.doc)[0]
        return el.getAttribute('network')


class VirtualDataCenter(APIType):
    content_type = 'application/vnd.vmware.vcloud.vdc+xml'

    @property
    def name(self):
        self.ensure_loaded()
        return self._get_attribute('//Vdc', 'name')

    @property
    def instantiate_url(self):
        self.ensure_loaded()
        return self._get_attribute('//Vdc/Link[@type="application/vnd.vmware.vcloud.instantiateVAppTemplateParams+xml"]',
                                   'href')

    @property
    def networks(self):
        self.ensure_loaded()
        return dict([(item.getAttribute('name'), item.getAttribute('href'))
                     for item in xpath.find('//Vdc/AvailableNetworks/Network', self.doc)])

    @property
    def vapps(self):
        self.ensure_loaded()
        return dict([(e.getAttribute("name"), 
                      VApp(self.session, e.getAttribute("href")))
                     for e in xpath.find("//Vdc/ResourceEntities/ResourceEntity[@type='%s']" % VApp.content_type, self.doc)])

    def _param_body(self, network_url, template, name):
        params = minidom.parseString("""<?xml version="1.0" encoding="UTF-8"?>
<InstantiateVAppTemplateParams
   xmlns="http://www.vmware.com/vcloud/v1.5"
   name=""
   deploy="true"
   powerOn="true"
   xsi:schemaLocation="http://schemas.dmtf.org/ovf/envelope/1 http://schemas.dmtf.org/ovf/envelope/1/dsp8023_1.1.0.xsd http://www.vmware.com/vcloud/v1.5 http://69.173.72.190/api/v1.5/schema/master.xsd"
   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
   xmlns:ovf="http://schemas.dmtf.org/ovf/envelope/1">
   <Description></Description>
   <InstantiationParams>
      <NetworkConfigSection>
         <ovf:Info>Configuration parameters for logical networks
         </ovf:Info>
         <NetworkConfig
            networkName="Imaging 194">
            <Configuration>
               <ParentNetwork
                  href="https://vcloud.example.com/api/network/54" />
               <FenceMode>bridged</FenceMode>
            </Configuration>
         </NetworkConfig>
      </NetworkConfigSection>
   </InstantiationParams>
   <Source
      href="https://vcloud.example.com/api/vAppTemplate/vappTemplate-111" />
</InstantiateVAppTemplateParams>""")
        root = xpath.find('//InstantiateVAppTemplateParams', params)[0]
        root.setAttribute('name', name)
        #root.setAttribute('description', "A machine to test the API")
        netconfig = xpath.find('//InstantiateVAppTemplateParams/InstantiationParams/NetworkConfigSection/NetworkConfig', params)[0]
        netconfig.setAttribute('networkName', template.network_name)
        net = xpath.find('//InstantiateVAppTemplateParams/InstantiationParams/NetworkConfigSection/NetworkConfig/Configuration/ParentNetwork', params)[0]
        net.setAttribute('href', network_url)
        source = xpath.find('//InstantiateVAppTemplateParams/Source', params)[0]
        source.setAttribute("href", template.url)
        return params.toxml()

    def deploy(self, item, name=None):
        assert len(self.networks) == 1
        network_url = self.networks.values()[0]
        template = item.template()
        #print
        #print 'TEMPLATE'
        #print template.toxml()
        name = name or item.name + '-' + random_string(6)
        body = self._param_body(network_url, template, name)
        #print
        #print 'PARAMS'
        #print body
        doc = self.session.post(self.instantiate_url, body,
                                body_type='application/vnd.vmware.vcloud.instantiateVAppTemplateParams+xml')
        return VApp(self.session, doc=doc)


class Task(APIType):
    content_type = 'application/vnd.vmware.vcloud.task+xml'

    @property
    def status(self):
        return self._get_attribute('//Task', 'status')

    @property
    def operation(self):
        return self._get_attribute('//Task', 'operation')

    @property
    def progress(self):
        return int(self._get_text_children('//Task/Progress'))

    def wait(self):
        while True:
            if self.status in ['success', 'error', 'canceled', 'aborted']:
                return self.status
            time.sleep(1)
            self.reload()

class OperationException(Exception):
    pass
        

class VApp(APIType):
    content_type = 'application/vnd.vmware.vcloud.vApp+xml'

    @property
    def name(self):
        self.ensure_loaded()
        return self._get_attribute('//VApp', 'name')
    
    @property
    def ip_address(self):
        self.ensure_loaded()
        return self._get_text_children("//VApp/Children/Vm/NetworkConnectionSection/NetworkConnection/IpAddress")

    @property
    def vm_status(self):
        self.ensure_loaded()
        status_code = self._get_attribute('//VApp/Children/Vm', 'status', None)
        return {None: "",
                "-1": "not_created",
                "0": "unresolved",
                "1": "resolved",
                "2": "deployed",
                "3": "suspended",
                "4": "powered_on",
                "5": "waiting_for_user_input",
                "6": "unknown_state",
                "7": "unrecognized_state",
                "8": "powered_off",
                "9": "inconsistent",
                "10": "children_differ",
                "11": "upload_initiated",
                "12": "upload_copying_contents",
                "13": "upload_disk_contents_pending",
                "14": "upload_quarantined",
                "15": "upload_quarantine_expired"}[status_code]

    @property
    def deployed(self):
        self.ensure_loaded()
        return {'true': True, 'false': False}[self._get_attribute('//VApp', 'deployed')]

    def undeploy(self):
        """Undeploy and power off the vApp."""
        self.ensure_loaded()
        url = self._get_attribute("//VApp/Link[@rel='undeploy']", 'href')
        body = """<?xml version="1.0" encoding="UTF-8"?>
                  <UndeployVAppParams
                     xmlns="http://www.vmware.com/vcloud/v1.5">
                     <UndeployPowerAction>default</UndeployPowerAction>
                  </UndeployVAppParams>"""
        doc = self.session.post(url, body, 
                                'application/vnd.vmware.vcloud.undeployVAppParams+xml')
        status = Task(self.session, doc=doc).wait()
        if status != "success":
            raise OperationException("Failed to undeploy. Status: %s" % status)
        self.reload()

    def delete(self):
        """Delete the vApp"""
        self.ensure_loaded()
        url = self._get_attribute("//VApp/Link[@rel='remove']", 'href')
        assert url
        doc = self.session.delete(url)
        status = Task(self.session, doc=doc).wait()
        if status != "success":
            raise OperationException("Failed to undeploy. Status: %s" % status)


def random_string(n):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(n))

if __name__ == '__main__':

    parser = OptionParser("""usage: %prog --help
       %prog list-catalogs
       %prog list-templates CATALOG
       %prog deploy CATALOG TEMPLATE
       %prog status [VM-NAME]
       %prog delete VM-NAME""")
    default_organization = "Imaging"
    default_user = getpass.getuser()
    default_base_url = "https://vcd.broadinstitute.org/api"
    parser.add_option("-b", "--base-url", default=default_base_url,
                      help="vCloud base URL [default: %s]" % default_base_url)
    parser.add_option("-o", "--organization", default=default_organization,
                      help="vCloud organization [default: %s]" % default_organization)
    parser.add_option("-p", "--password",
                      help="vCloud password [default: ask]")
    parser.add_option("-u", "--user", default=default_user,
                      help="vCloud username [default: %s]" % default_user)
    options, args = parser.parse_args()

    def die(message):
        print >>sys.stderr, "%s: %s" % (os.path.basename(sys.argv[0]), message)
        sys.exit(1)

    def make_session():
        username = options.user
        organization = options.organization
        password = options.password or getpass.getpass()
        return Session(options.base_url, username, organization, password)

    def help():
        parser.print_usage()
        sys.exit(1)

    def command_list_catalogs():
        if len(args) != 1:
            help()
        session = make_session()
        org = session.organization()
        for name in org.catalogs.keys():
            print name

    def _get_catalog(catalog_name):
        session = make_session()
        org = session.organization()
        catalog_name = args[1]
        catalogs = org.catalogs
        if catalog_name not in catalogs:
            die("Can't find catalog: %s" % catalog_name)
        return catalogs[catalog_name], org

    def command_list_templates():
        if len(args) != 2:
            help()
        catalog, org = _get_catalog(args[1])
        template_names = catalog.keys()
        for name in template_names:
            print name

    def command_deploy():
        if len(args) != 3:
            help()
        catalog, org = _get_catalog(args[1])
        item = catalog[args[2]]
        vdc = org.virtual_data_center()
        vapp = vdc.deploy(item, name=os.environ.get('BUILD_TAG'))
        print vapp.name

    def command_status():
        if len(args) == 1:
            session = make_session()
            org = session.organization()
            vdc = org.virtual_data_center()
            vapps = vdc.vapps.values()
        elif len(args) == 2:
            vapp_name = args[1]
            session = make_session()
            org = session.organization()
            vdc = org.virtual_data_center()
            d = vdc.vapps
            if vapp_name not in d:
                die("Can't find VApp: %s" % vapp_name)
            vapps = [d[vapp_name]]
        else:
            help()
        print '\t'.join(["NAME", "STATUS", "IP_ADDRESS"])
        for vapp in vapps:
            print '\t'.join([vapp.name, vapp.vm_status, 
                             vapp.ip_address or "-"])

    def command_delete():
        vapp_name = args[1]
        session = make_session()
        org = session.organization()
        vdc = org.virtual_data_center()
        d = vdc.vapps
        if vapp_name not in d:
            die("Can't find VApp: %s" % vapp_name)
        vapp = d[vapp_name]
        try:
            if vapp.deployed:
                vapp.undeploy()
            vapp.delete()
        except OperationException, e:
            print e
            sys.exit(1)

    def command_experiment():
        session = make_session()
        org = session.organization()
        catalog = org.catalog()
        #print catalog.items
        #item = catalog['CentoOS 6 x64-copy-bdb911dd-cbd1-41ed-a756-42527ffe16b4']
        #template = item.template()
        vdc = org.virtual_data_center()
        print vdc.toxml()
        #print 
        #print '*** NETWORKS'
        #print vdc.networks

        #vapp = vdc.deploy(item)
        #print vapp.toxml()

    if len(args) < 1:
        help()
    command = args[0]
    if command == 'list-catalogs':
        command_list_catalogs()
    elif command == 'list-templates':
        command_list_templates()
    elif command == 'deploy':
        command_deploy()
    elif command == 'status':
        command_status()
    elif command == 'delete':
        command_delete()
    elif command == 'experiment':
        command_experiment()
    else:
        help()
