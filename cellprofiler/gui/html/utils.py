import docutils


# https://wiki.python.org/moin/reStructuredText
def rst_to_html_fragment(source):
    parts = docutils.core.publish_parts(source=source, writer_name="html")

    return parts["body_pre_docinfo"] + parts["fragment"]