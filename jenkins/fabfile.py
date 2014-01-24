# Fabric file that connects to a fresh virtual machine, sets up build
# dependencies, runs the build, and copies out the product and any
# error messages.
#
# During development, run with the IP address of the virtual machine
# in the -H parameter. Example: fab -H 192.168.194.177 build
#
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

from fabric.api import env, settings, run, put, get, local
from fabric.decorators import with_settings

env.user = "cpbuild"

@with_settings(user="root")
def set_up_user(username):
    home = '/home/' + username
    d = dict(home=home, username=username)
    run("""test -d {home} || adduser {username}""".format(**d))
    run("""test -d {home}/.ssh || sudo -u {username} mkdir -m 700 {home}/.ssh""".format(**d))
    put("id_rsa.pub", "{home}/.ssh/authorized_keys".format(**d), mode=0600)
    run("""chown {username}:{username} {home}/.ssh/authorized_keys""".format(**d))
    run("""echo '{username}	ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers""".format(**d))

def build():
    set_up_user("cpbuild")
    local("tar cpf workspace.tar --exclude workspace.tar ..")
    put("workspace.tar")
    put("build_cellprofiler.sh", "~", mode=0755)
    run("./build_cellprofiler.sh")
    get("cellprofiler.tar.gz", "cellprofiler.tar.gz")

def deploy():
    with settings(user="root"):
        run("yum -y install gtk2-devel mesa-libGL mesa-libGL-devel blas atlas lapack blas-devel atlas-devel lapack-devel xorg-x11-xauth* xorg-x11-xkb-utils* qt-devel openssl openssl-devel xclock *Xvfb* svn libXtst")
        put("cellprofiler.tar.gz")
        run("tar -C / -xzf cellprofiler.tar.gz")

def test():
    set_up_user("johndoe")
    deploy()
    with settings(user="johndoe"):
        run("/usr/CellProfiler/CellProfiler/shortcuts/cellprofiler -t")

def build_wxpython_rpm():
    put("wxPython2.8-2.8.12.1-1.src.rpm", "~")
    put("wxpython_rpm.diff", "~")
    put("build_wxpython_rpm.sh", "~", mode=0755)
    run("./build_wxpython_rpm.sh")
    for fn in ['wxPython2.8-devel-gtk2-unicode-2.8.12.1-1_py2.6.x86_64.rpm',
               'wxPython2.8-gtk2-unicode-2.8.12.1-1_py2.6.x86_64.rpm',
               'wxPython-common-gtk2-unicode-2.8.12.1-1_py2.6.x86_64.rpm']:
        get("~/rpmbuild/RPMS/x86_64/%s" % fn, fn)

