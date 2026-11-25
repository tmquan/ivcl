"""Link-preview tags (Open Graph, Twitter card) for the pages listed in PAGES.

A local Sphinx extension, named in _config.yml, so that every build of the book writes the tags
into the <head> of these pages; nothing has to be added to the built HTML by hand.
"""
SITE = 'https://tmquan.github.io/ivcl/'
SITE_NAME = 'Intelligent Visual Computing'

PAGES = {
    'courses/fme/index': dict(
        title='BJT and MOSFET Circuits',
        description='One transistor in eight roles, its impedances and gains, and forty-two circuits of two '
                    'transistors, each drawn as a BJT version and as a MOSFET version.',
        image='_images/fme_gallery.png', width=1200, height=630,
        alt='Five circuits drawn twice, with BJTs and with MOSFETs: current mirror, cascode, differential pair, '
            'active load, cross-coupled pair.'),
}


def _tags(app, pagename, templatename, context, doctree):
    page = PAGES.get(pagename)
    if not page:
        return
    from html import escape
    e = lambda s: escape(str(s), quote=True)
    image = SITE + page['image']
    tags = [('name', 'description', page['description']),
            ('property', 'og:type', 'article'),
            ('property', 'og:site_name', SITE_NAME),
            ('property', 'og:title', page['title']),
            ('property', 'og:description', page['description']),
            ('property', 'og:url', SITE + pagename + '.html'),
            ('property', 'og:image', image),
            ('property', 'og:image:type', 'image/png'),
            ('property', 'og:image:width', page['width']),
            ('property', 'og:image:height', page['height']),
            ('property', 'og:image:alt', page['alt']),
            ('name', 'twitter:card', 'summary_large_image'),
            ('name', 'twitter:image', image)]
    context['metatags'] = context.get('metatags', '') + ''.join(
        '\n    <meta %s="%s" content="%s" />' % (kind, key, e(value)) for kind, key, value in tags)


def setup(app):
    app.connect('html-page-context', _tags)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
