import datetime
import requests
import wx
import re

from packaging.version import Version, VERSION_PATTERN

from cellprofiler_core.preferences import get_check_update, set_check_update, get_check_update_bool
from cellprofiler import __version__ as current_version


def check_update(parent, force=False):
    if not force and not check_date():
        return
    try:
        response = requests.get("https://api.github.com/repos/cellprofiler/cellprofiler/releases/latest", timeout=0.25)
    except:
        response = False
        message = "CellProfiler was unable to connect to GitHub to check for updates"
    if response:
        status = response.status_code
        response = response.json()
        if status == 200 and 'name' in response:
            latest_version = response['name'][1:]
            if not re.match(VERSION_PATTERN, latest_version, re.IGNORECASE | re.VERBOSE):
                message = f"Unable to parse version number from GitHub: {latest_version}"
            elif Version(current_version) < Version(latest_version):
                body_text = response['body']
                if len(body_text) > 1000:
                    body_text = body_text[:1000] + "..."
                elif len(body_text) == 0:
                    body_text = "No information available"
                show_message(parent, latest_version, body_text)
                return
            else:
                message = "CellProfiler is up-to-date"
                if get_check_update() != "Disabled":
                    set_check_update(datetime.date.today().strftime("%Y%m%d"))
                if not force:
                    return
        elif status == 200:
            message = "Unable to read data from GitHub, API may have changed."
        else:
            message = "Invalid response from GitHub server, site may be down."
    if force:
        dlg = wx.MessageDialog(
            parent,
            message,
            caption="Check for updates",
            style=wx.ICON_INFORMATION | wx.OK,
        )
        dlg.ShowModal()
    else:
        print(message)


def show_message(parent, version, blurb):
    message = f"""A new CellProfiler release is available:\n\nVersion {version}\n
Would you like to visit the download page?"""
    dlg = wx.RichMessageDialog(
        parent,
        message,
        caption="CellProfiler Update Available",
        style=wx.YES_NO | wx.CENTRE | wx.ICON_INFORMATION,
    )
    dlg.ShowDetailedText(f"Release Notes:\n{blurb}")
    dlg.ShowCheckBox("Check for updates on startup", checked=get_check_update_bool())
    response = dlg.ShowModal()
    if response == wx.ID_YES:
        wx.LaunchDefaultBrowser("https://cellprofiler.org/releases")
    if not dlg.IsCheckBoxChecked():
        set_check_update("Disabled")
    else:
        set_check_update(datetime.date.today().strftime("%Y%m%d"))


def check_date():
    last_checked = get_check_update()
    if last_checked == "Disabled":
        # Updating is disabled
        return False
    elif last_checked == "Never":
        return True
    today = datetime.date.today()
    last_checked = datetime.datetime.strptime(last_checked, "%Y%m%d").date()
    if (last_checked - today).days >= 7:
        return True
    else:
        return False
