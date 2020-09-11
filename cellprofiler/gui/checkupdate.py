from cellprofiler import __version__ as current_version

import requests
import wx


def check_update(parent):
    try:
        response = requests.get("https://api.github.com/repos/cellprofiler/cellprofiler/releases/latest")
    except:
        print("CellProfiler was unable to connect to GitHub to check for updates")
        return
    response = response.json()
    if 'name' in response:
        latest_version = response['name'][1:]
        if current_version < latest_version:
            print("You'd better update dude")
            body_text = response['body']
            if len(body_text) > 100:
                body_text = body_text[:100] + "..."
            elif len(body_text) == 0:
                body_text = "No information available"
            show_message(parent, latest_version, body_text)
        else:
            print("CellProfiler is up-to-date")
    else:
        print("Unable to read data from GitHub, API may have changed.")


def show_message(parent, version, blurb):
    message = f"""A new version of CellProfiler is available:\nVersion {version}\n
Would you like to visit the download page?"""
    dlg = wx.RichMessageDialog(
        parent,
        message,
        caption="CellProfiler Update Available",
        style=wx.YES_NO | wx.CENTRE | wx.ICON_INFORMATION,
    )
    dlg.ShowDetailedText(f"Release Notes:\n{blurb}")
    dlg.ShowCheckBox("Check for updates on startup", checked=True)
    response = dlg.ShowModal()
    if response == wx.ID_YES:
        wx.LaunchDefaultBrowser("https://cellprofiler.org/releases")
    else:
        return
