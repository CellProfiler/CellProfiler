import glob
import os.path
import logging

import jinja2
import importlib.resources

import wx
import wx.html2
from cellprofiler_core.preferences import set_startup_blurb

import cellprofiler.gui.help.content
import cellprofiler.gui.html
import cellprofiler.gui.html.utils
import cellprofiler.gui.utilities.icon
from cellprofiler.gui.help.content import image_resource_dataUrl

logger = logging.getLogger(__name__)

class WelcomeFrame(wx.Frame):
    def __init__(self, parent):
        super(WelcomeFrame, self).__init__(
            parent,
            name="WelcomeScreenFrame",
            size=(640, 480),
            title="Welcome to CellProfiler",
        )

        self.SetSizer(wx.BoxSizer())

        self.SetIcon(cellprofiler.gui.utilities.icon.get_cp_icon())

        self.Bind(wx.EVT_CLOSE, self.__on_close)

        # Set to first supportable backend
        backends = [
            (wx.html2.WebViewBackendIE, 'WebViewBackendIE'),
            (wx.html2.WebViewBackendWebKit, 'WebViewBackendWebKit'),
            (wx.html2.WebViewBackendDefault, 'WebViewBackendDefault'),
        ]
        major, minor = list(map(int, wx.__version__.split('.')[0:2]))
        if major >= 4 and minor >= 2:
            backends = [(wx.html2.WebViewBackendEdge, 'WebViewBackendEdge')] + backends

        # Find an available backend
        backend = None
        for id, name in backends:
            available = wx.html2.WebView.IsBackendAvailable(id)
            logger.debug("WebView backend 'wx.html2.{}' availability: {}\n".format(name, available))
            if available and backend is None:
                backend = id
        logger.debug("Using WebView backend: '{}'\n".format(str(backend, 'ascii')))

        self.content = wx.html2.WebView.New(self, backend=backend)
        self.content.EnableContextMenu(False)
        self.__display_welcome()

        self.content.Bind(wx.html2.EVT_WEBVIEW_NAVIGATING, self._on_navigate)
        # self.content.Bind(wx.html2.EVT_WEBVIEW_LOADED, self._on_loaded)

        self.GetSizer().Add(self.content, 1, wx.EXPAND)

        self.Layout()

        self.href_to_help = {
            "configureimages": "html/configure_images_help.html",
            "exportingresults": "html/exporting_results_help.html",
            "identifyfeatures": "html/identify_features_help.html",
            "inapphelp": "html/in_app_help.html",
            "makingmeasurements": "html/making_measurements_help.html",
            "runningpipeline": "html/running_pipeline_help.html",
            "testmode": "html/test_mode_help.html",
            "selectingimages": "html/selecting_images_help.html",
            "gettingstarted": "html/getting_started.html",
            "welcome": "html/welcome.html",
        }

    def __display_welcome(self):
        with open(os.path.join(os.path.dirname(__file__), "html/welcome.html")) as fp:
            template = jinja2.Template(fp.read())
            self.content.SetPage(
                html=template.render(
                    MANUAL_URL=cellprofiler.gui.help.content.MANUAL_URL,
                    WELCOME_MANUAL=image_resource_dataUrl("welcome_manual.png"),
                    WELCOME_FORUM=image_resource_dataUrl("welcome_forum.png"),
                    WELCOME_PIPELINE=image_resource_dataUrl("welcome_pipeline.png"),
                    WELCOME_TUTORIALS=image_resource_dataUrl("welcome_tutorial.png"),
                    WELCOME_EXAMPLES=image_resource_dataUrl("welcome_examples.png"),
                    WELCOME_START=image_resource_dataUrl("welcome_start.png"),
                    WELCOME_HELP=image_resource_dataUrl("welcome_help.png"),
                    WELCOME_NEW=image_resource_dataUrl("welcome_new.png"),
                ),
                baseUrl="file:welcome",
            )

    @staticmethod
    def __on_close(event):
        event.EventObject.Hide()

        event.Veto()

    def _on_navigate(self, event):
        href = event.URL
        if href.startswith("help:"):
            self.__display_help(href[5:])
        elif href.startswith("file:"):
            # Ignore incorrect navigation calls on Mac
            return
        elif href.startswith("loadexample:"):
            self.__load_example_pipeline(href[12:])
            return
        elif href.startswith("pref:"):
            self.__set_startup_blurb()
        elif href.startswith("http"):
            wx.LaunchDefaultBrowser(href)
            # prevent loading in screen
            event.Veto()

    def __display_help(self, href):
        if href == "welcome":
            self.__display_welcome()
            return
        if href in self.href_to_help:
            html_path = self.href_to_help[href]
        else:
            html_path = href
        with open(os.path.join(os.path.dirname(__file__), html_path)) as fp:
            template = jinja2.Template(fp.read())
            self.content.SetPage(
                html=template.render(
                    GO_BACK="""<p>Go <a href=help:gettingstarted>back</a> to the previous screen.</p>""",
                    GO_HOME="""<p>Go <a href=help:welcome>back</a> to the welcome screen.</p>""",
                    MODULE_HELP_BUTTON=image_resource_dataUrl(
                        cellprofiler.gui.help.content.MODULE_HELP_BUTTON
                    ),
                    MODULE_ADD_BUTTON=image_resource_dataUrl(
                        cellprofiler.gui.help.content.MODULE_ADD_BUTTON
                    ),
                    ANALYZE_BUTTON=image_resource_dataUrl(
                        cellprofiler.gui.help.content.ANALYZE_IMAGE_BUTTON
                    ),
                    PAUSE_BUTTON=image_resource_dataUrl(
                        cellprofiler.gui.help.content.PAUSE_ANALYSIS_BUTTON
                    ),
                    PAUSE_BUTTON_DIM=image_resource_dataUrl(
                        cellprofiler.gui.help.content.INACTIVE_PAUSE_BUTTON
                    ),
                    STEP_BUTTON_DIM=image_resource_dataUrl(
                        cellprofiler.gui.help.content.INACTIVE_STEP_BUTTON
                    ),
                    STOP_BUTTON=image_resource_dataUrl(
                        cellprofiler.gui.help.content.STOP_ANALYSIS_BUTTON
                    ),
                    IMAGE_OBJECT_DATAFLOW=image_resource_dataUrl("image_to_object_dataflow.png"),
                ),
                baseUrl=html_path,
            )

    @staticmethod
    def __load_example_pipeline(example_name):
        example_dir =  os.path.join(
            importlib.resources.files("cellprofiler"),
            "data",
            "examples",
            example_name
        )

        pipeline_pathname = os.path.join(
            example_dir, "{:s}.cppipe".format(example_name)
        )

        images_dir = os.path.join(example_dir, "images")

        try:

            def load(pathname=pipeline_pathname):
                pipeline = wx.GetApp().frame.pipeline
                pipeline.load(pathname)
                pipeline.add_pathnames_to_file_list(
                    glob.glob(os.path.join(images_dir, "*"))
                )

                wx.MessageBox(
                    'Now that you have loaded an example pipeline, press the "Analyze images" button to access and'
                    " process a small image set from the CellProfiler website so you can see how CellProfiler works.",
                    "",
                    wx.ICON_INFORMATION,
                )

            wx.CallAfter(load)
        except:
            wx.MessageBox(
                "CellProfiler was unable to load {}".format(pipeline_pathname),
                "Error loading pipeline",
                style=wx.OK | wx.ICON_ERROR,
            )

    def __set_startup_blurb(self):
        set_startup_blurb(False)
        wx.MessageBox(
            'This page can be accessed from "Help --> Show Welcome Screen" at any time.\n'
            "",
            "Welcome screen will no longer display on startup.",
            wx.ICON_INFORMATION,
        )

        self.Close()
