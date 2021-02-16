install() {
  PYDEST="$PREFIX/lib/python"
  PYTHONPATH="$PYDEST:$PYTHONPATH" \
     eval python setup.py install --single-version-externally-managed --record record.txt  $PYSETUP_INSTALL_OPTIONS
  install_ups
}
