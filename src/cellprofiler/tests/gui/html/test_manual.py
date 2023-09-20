# # coding=utf-8
# """Tests for cellprofiler.gui.html.manual"""
#
# import cellprofiler.gui.html.manual
# import cellprofiler_core.preferences
# import os
# import re
# import tempfile
# import traceback
# import unittest
#
#
# class TestManual(unittest.TestCase):
#     def setUp(self):
#         self.temp_dir = tempfile.mkdtemp()
#         from cellprofiler_core.setting import Text
#
#         def make_command_choice(self, label, doc):
#             return Text(label, "None", doc=doc)
#
#         def populate_language_dictionary(self):
#             self.language_dictionary = dict(swedish="SwedishEngine")
#
#     def tearDown(self):
#         for root, dirnames, filenames in os.walk(self.temp_dir, False):
#             for x in filenames:
#                 os.remove(os.path.join(root, x))
#             for x in dirnames:
#                 os.remove(os.path.join(root, x))
#         os.rmdir(self.temp_dir)
#
#     def test_01_01_output_module_html(self):
#         from cellprofiler.modules import get_module_names, instantiate_module
#         cellprofiler.gui.html.manual.output_module_html(self.temp_dir)
#         for module_name in sorted(get_module_names()):
#             fd = None
#             try:
#                 fd = open(os.path.join(self.temp_dir, module_name + ".html"))
#             except:
#                 module = instantiate_module(module_name)
#                 location = os.path.split(
#                         module.create_settings.im_func.func_code.co_filename)[0]
#                 if location == cellprofiler_core.preferences.get_plugin_directory():
#                     continue
#                 traceback.print_exc()
#                 self.assert_("Failed to open %s.html" % module_name)
#             data = fd.read()
#             fd.close()
#
#             #
#             # Make sure that some nesting rules are obeyed.
#             #
#             tags_we_care_about = ("i", "b", "ul", "ol", "li", "table", "tr", "td", "th",
#                                   "h1", "h2", "h3", "html", "head", "body")
#             pattern = r"<\s*([a-zA-Z0-9]+).[^>]*>"
#             anti_pattern = r"</\s*([a-zA-Z0-9]+)[^>]*>"
#             d = {}
#             anti_d = {}
#             COUNT = 0
#             LIST = 1
#             for tag in tags_we_care_about:
#                 for dd in (d, anti_d):
#                     dd[tag] = [0, []]
#
#             for p, dd in ((pattern, d),
#                           (anti_pattern, anti_d)):
#                 pos = 0
#                 while True:
#                     m = re.search(p, data[pos:])
#                     if m is None:
#                         break
#                     tag = m.groups()[0].lower()
#                     pos = pos + m.start(1) + 1
#                     if dd.has_key(tag):
#                         dd[tag][COUNT] += 1
#                         dd[tag][LIST].append(pos)
#             #
#             # Check table nesting rules
#             #
#             T_TABLE = 0
#             T_ANTI_TABLE = 1
#             T_TR = 2
#             T_ANTI_TR = 3
#             T_TH = 4
#             T_ANTI_TH = 5
#             T_TD = 6
#             T_ANTI_TD = 7
#             T_UL = 8
#             T_ANTI_UL = 9
#             T_OL = 10
#             T_ANTI_OL = 11
#             T_LI = 12
#             T_ANTI_LI = 13
#             T_I = 14
#             T_ANTI_I = 15
#             T_B = 16
#             T_ANTI_B = 17
#             tokens = []
#             for tag, token, anti_token in (
#                     ('table', T_TABLE, T_ANTI_TABLE),
#                     ('tr', T_TR, T_ANTI_TR),
#                     ('td', T_TD, T_ANTI_TD),
#                     ('th', T_TH, T_ANTI_TH),
#                     ('ul', T_UL, T_ANTI_UL),
#                     ('ol', T_OL, T_ANTI_OL),
#                     ('li', T_LI, T_ANTI_LI),
#                     ('i', T_I, T_ANTI_I),
#                     ('b', T_B, T_ANTI_B)
#             ):
#                 tokens += [(pos, token) for pos in d[tag][LIST]]
#                 tokens += [(pos, anti_token) for pos in anti_d[tag][LIST]]
#
#             tokens = sorted(tokens)
#             S_INIT = 0
#             S_AFTER_TABLE = 1
#             S_AFTER_TR = 2
#             S_AFTER_TD = 3
#             S_AFTER_TH = 4
#             S_AFTER_OL = 5
#             S_AFTER_UL = 6
#             S_AFTER_LI = 7
#             S_AFTER_I = 8
#             S_AFTER_B = 9
#
#             state_transitions = {
#                 S_INIT: {T_TABLE: S_AFTER_TABLE,
#                          T_OL: S_AFTER_OL,
#                          T_UL: S_AFTER_UL,
#                          T_I: S_AFTER_I,
#                          T_B: S_AFTER_B},
#                 S_AFTER_TABLE: {
#                     T_ANTI_TABLE: S_INIT,
#                     T_TR: S_AFTER_TR
#                 },
#                 S_AFTER_TR: {
#                     T_ANTI_TR: S_INIT,
#                     T_TD: S_AFTER_TD,
#                     T_TH: S_AFTER_TH
#                 },
#                 S_AFTER_TD: {
#                     T_TABLE: S_AFTER_TABLE,
#                     T_OL: S_AFTER_OL,
#                     T_UL: S_AFTER_UL,
#                     T_B: S_AFTER_B,
#                     T_I: S_AFTER_I,
#                     T_ANTI_TD: S_INIT
#                 },
#                 S_AFTER_TH: {
#                     T_TABLE: S_AFTER_TABLE,
#                     T_OL: S_AFTER_OL,
#                     T_UL: S_AFTER_UL,
#                     T_B: S_AFTER_B,
#                     T_I: S_AFTER_I,
#                     T_ANTI_TH: S_INIT
#                 },
#                 S_AFTER_OL: {
#                     T_LI: S_AFTER_LI,
#                     T_ANTI_OL: S_INIT
#                 },
#                 S_AFTER_UL: {
#                     T_LI: S_AFTER_LI,
#                     T_ANTI_UL: S_INIT
#                 },
#                 S_AFTER_LI: {
#                     T_ANTI_LI: S_INIT,
#                     T_TABLE: S_AFTER_TABLE,
#                     T_OL: S_AFTER_OL,
#                     T_UL: S_AFTER_UL,
#                     T_B: S_AFTER_B,
#                     T_I: S_AFTER_I
#                 },
#                 S_AFTER_I: {
#                     T_ANTI_I: S_INIT,
#                     T_I: S_AFTER_I,  # Stupid but legal <i><i>Foo</i></i>
#                     T_B: S_AFTER_B,
#                     T_TABLE: S_AFTER_TABLE,
#                     T_OL: S_AFTER_OL,
#                     T_UL: S_AFTER_UL
#                 },
#                 S_AFTER_B: {
#                     T_ANTI_B: S_INIT,
#                     T_B: S_AFTER_B,
#                     T_I: S_AFTER_I,
#                     T_TABLE: S_AFTER_TABLE,
#                     T_OL: S_AFTER_OL,
#                     T_UL: S_AFTER_UL
#                 }
#             }
#             state = []
#
#             for pos, token in tokens:
#                 self.assertTrue(
#                         len(state) >= 0,
#                         "Error in %s near position %d (%s)" %
#                         (module_name, pos, data[max(0, pos - 30):
#                         max(pos + 30, len(data))])
#                 )
#                 top_state, start_pos = (S_INIT, 0) if len(state) == 0 else state[-1]
#
#                 self.assertTrue(
#                         state_transitions[top_state].has_key(token),
#                         "Nesting error in %s near position %d (%s)" %
#                         (module_name, pos, data[max(0, pos - 50):pos] + "^^^" +
#                          data[pos:min(pos + 50, len(data))]))
#                 next_state = state_transitions[top_state][token]
#                 if next_state == S_INIT:
#                     state.pop()
#                 else:
#                     state.append((next_state, pos))
#             if len(state) > 0:
#                 self.assertEqual(
#                         len(state), 0,
#                         "Couldn't find last closing tag in %s. Last tag position = %d (%s)" %
#                         (module_name, state[-1][1], data[(state[-1][1] - 30):
#                         (state[-1][1] + 30)]))
#             #
#             # Check begin/end tag counts
#             #
#             for tag in tags_we_care_about:
#                 if d.has_key(tag):
#                     self.assertTrue(anti_d.has_key(tag),
#                                     "Missing closing </%s> tag in %s" %
#                                     (tag, module_name))
#                     self.assertEqual(
#                             d[tag][COUNT], anti_d[tag][COUNT],
#                             "Found %d <%s>, != %d </%s> in %s" %
#                             (d[tag][COUNT], tag,
#                              anti_d[tag][COUNT], tag, module_name))
#                 else:
#                     self.assertFalse(anti_d.has_key(tag),
#                                      "Missing opening <%s> tag in %s" %
#                                      (tag, module_name))
