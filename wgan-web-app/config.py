"""Database configuration, read from environment variables.

Set these before running the app, for example:

    set MYSQL_HOST=127.0.0.1
    set MYSQL_PORT=3306
    set MYSQL_USER=root
    set MYSQL_PASSWORD=your-password
    set MYSQL_DATABASE=C4

The dict name ``DtatBaseConf`` is kept for backwards compatibility with
``main.py``, which imports it under that name.
"""
import os

DtatBaseConf = {
    'HOSTNAME': os.environ.get('MYSQL_HOST', '127.0.0.1'),
    'PORT': os.environ.get('MYSQL_PORT', '3306'),
    'USERNAME': os.environ.get('MYSQL_USER', 'root'),
    'PASSWORD': os.environ.get('MYSQL_PASSWORD', ''),
    'DATABASE': os.environ.get('MYSQL_DATABASE', 'C4'),
}
