# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[("Assets", "Assets"), ("Models", "Models")],
    hiddenimports=[
        'numpy', 
        'numpy.core._multiarray_umath',
        'pypdf',
        'pdf2image',
        'PIL',
        'PIL.Image',
        'pyctcdecode',
        'pyewts',
        'platformdirs'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(
    a.pure,
    a.zipped_data
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)

app = BUNDLE(coll,
             name='BDRC-MonlamAI Tibetan OCR.app',
             icon="logo.icns",
             bundle_identifier=None)
