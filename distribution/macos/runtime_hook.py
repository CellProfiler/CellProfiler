import os
import sys

os.environ["JAVA_HOME"] = os.environ.get(
    "CP_JAVA_HOME", os.path.join(sys._MEIPASS, "jre")
)
os.environ.setdefault(
    "LIBMYSQL_PLUGIN_DIR", os.path.join(sys._MEIPASS, "mysql/plugin")
)
