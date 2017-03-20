# -*- mode: python -*-

block_cipher = None

datas = [
    ('cellprofiler/data/*', 'cellprofiler/data')
]

a = Analysis(
    [
        'CellProfiler.py'
    ],
    pathex=['C:\\Users\\Public\\CellProfiler'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "cellprofiler.modules.crop",
        "cellprofiler.modules.align",
        "cellprofiler.modules.activecontourmodel",
        "cellprofiler.modules.blobdetection",
        "cellprofiler.modules.calculateimageoverlap",
        "cellprofiler.modules.calculatemath",
        "cellprofiler.modules.calculatestatistics",
        "cellprofiler.modules.classifyobjects",
        "cellprofiler.modules.closing",
        "cellprofiler.modules.colortogray",
        "cellprofiler.modules.convertobjectstoimage",
        "cellprofiler.modules.correctilluminationcalculate",
        "cellprofiler.modules.correctilluminationapply",
        "cellprofiler.modules.createbatchfiles",
        "cellprofiler.modules.definegrid",
        "cellprofiler.modules.dilation",
        "cellprofiler.modules.displaydataonimage",
        "cellprofiler.modules.displaydensityplot",
        "cellprofiler.modules.displayhistogram",
        "cellprofiler.modules.displayplatemap",
        "cellprofiler.modules.displayscatterplot",
        "cellprofiler.modules.editobjectsmanually",
        "cellprofiler.modules.edgedetection",
        "cellprofiler.modules.enhanceedges",
        "cellprofiler.modules.enhanceorsuppressfeatures",
        "cellprofiler.modules.erosion",
        "cellprofiler.modules.expandorshrinkobjects",
        "cellprofiler.modules.exporttodatabase",
        "cellprofiler.modules.exporttospreadsheet",
        "cellprofiler.modules.filterobjects",
        "cellprofiler.modules.flagimage",
        "cellprofiler.modules.flipandrotate",
        "cellprofiler.modules.gammacorrection",
        "cellprofiler.modules.gaussianfilter",
        "cellprofiler.modules.graytocolor",
        "cellprofiler.modules.histogramequalization",
        "cellprofiler.modules.identifydeadworms",
        "cellprofiler.modules.identifyobjectsingrid",
        "cellprofiler.modules.identifyobjectsmanually",
        "cellprofiler.modules.identifyprimaryobjects",
        "cellprofiler.modules.identifysecondaryobjects",
        "cellprofiler.modules.identifytertiaryobjects",
        "cellprofiler.modules.imagegradient",
        "cellprofiler.modules.imagemath",
        "cellprofiler.modules.invertforprinting",
        "cellprofiler.modules.labelimages",
        "cellprofiler.modules.laplacianofgaussian",
        "cellprofiler.modules.makeprojection",
        "cellprofiler.modules.maskimage",
        "cellprofiler.modules.maskobjects",
        "cellprofiler.modules.measurecorrelation",
        "cellprofiler.modules.measuregranularity",
        "cellprofiler.modules.measureimageareaoccupied",
        "cellprofiler.modules.measureimageintensity",
        "cellprofiler.modules.measureimagequality",
        "cellprofiler.modules.measureobjectintensity",
        "cellprofiler.modules.measureobjectsizeshape",
        "cellprofiler.modules.measureobjectneighbors",
        "cellprofiler.modules.measureobjectintensitydistribution",
        "cellprofiler.modules.measureneurons",
        "cellprofiler.modules.measuretexture",
        "cellprofiler.modules.medianfilter",
        "cellprofiler.modules.medialaxis",
        "cellprofiler.modules.mergeoutputfiles",
        "cellprofiler.modules.morph",
        "cellprofiler.modules.opening",
        "cellprofiler.modules.noisereduction",
        "cellprofiler.modules.overlayoutlines",
        "cellprofiler.modules.randomwalkeralgorithm",
        "cellprofiler.modules.relateobjects",
        "cellprofiler.modules.reassignobjectnumbers",
        "cellprofiler.modules.removeholes",
        "cellprofiler.modules.removeobjects",
        "cellprofiler.modules.rescaleintensity",
        "cellprofiler.modules.resize",
        "cellprofiler.modules.save",
        "cellprofiler.modules.saveimages",
        "cellprofiler.modules.smooth",
        "cellprofiler.modules.straightenworms",
        "cellprofiler.modules.trackobjects",
        "cellprofiler.modules.tile",
        "cellprofiler.modules.unmixcolors",
        "cellprofiler.modules.untangleworms",
        "cellprofiler.modules.watershed",
        "prokaryote",
        "zmq",
        "zmq.backend.cython"
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "zmq.libzmq"
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)

a.binaries = [x for x in a.binaries if not x[0].startswith("libzmq.pyd")]

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='CellProfiler',
          debug=False,
          strip=False,
          upx=True,
          console=True)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='CellProfiler')

# cp C:\Python27\Lib\site-packages\zmq\libzmq.pyd .\dist\CellProfiler\
# cp -R C:\Python27\Lib\site-packages\prokaryote .\dist\CellProfiler\
# cp -R C:\Python27\Lib\site-packages\javabridge\ .\dist\CellProfiler\

# Traceback (most recent call last):
#   File "cellprofiler\gui\pipelinecontroller.py", line 2847, in do_step
#     self.__pipeline.run_module(module, workspace)
#   File "cellprofiler\pipeline.py", line 1997, in run_module
#     module.run(workspace)
#   File "cellprofiler\modules\identifysecondaryobjects.py", line 479, in run
#     (labels_in[0, :], labels_in[-1, :], labels_in[:, 0], labels_in[:, -1]))
#   File "site-packages\numpy\core\shape_base.py", line 288, in hstack
# ValueError: all the input array dimensions except for the concatenation axis must match exactly
