# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['TagMetaUpdater.py'],
    pathex=[
        'TagMetaUpdater', 
        'venv\\lib\\site-packages\\huggingface_hub'
        ],
    binaries=[
        ('venv\\Lib\\site-packages\\pyexiv2', 'pyexiv2'),
        ('venv\\Lib\\site-packages\\onnxruntime', 'onnxruntime')
        ],
    datas=[
        ('config.ini', '.'),
        ('readme.md', '.')
        ],
    hiddenimports=[
        'onnxruntime', 
        'huggingface_hub'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TagMetaUpdater',
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
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TagMetaUpdater',
)
