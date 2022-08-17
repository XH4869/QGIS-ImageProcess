# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ImageProcess
                                 A QGIS plugin
 A plugin for remote sensing images processing
                             -------------------
        begin                : 2019-06-05
        copyright            : (C) 2019 by Team Remote Control
        email                : 2016302590109@whu.edu.cn
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load ImageProcess class from file ImageProcess.

    :param iface: A QGIS interface instance.
    :type iface: QgisInterface
    """
    #
    from .ImageProcess import ImageProcess
    return ImageProcess(iface)
