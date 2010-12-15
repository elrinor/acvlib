/* This file is part of ACVLib, an ArX Computer Vision Library.
 *
 * Copyright (C) 2009-2010 Alexander Fokin <apfokin@gmail.com>
 *
 * ACVLib is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * ACVLib is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License 
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ACVLib. If not, see <http://www.gnu.org/licenses/>. 
 * 
 * $Id$ */
#ifndef ACV_CONFIG_H
#define ACV_CONFIG_H

#ifdef ACV_DEBUG
#  include <iostream>
#  define ACV_DEBUG_PRINT(ARGS) ::std::cout << ARGS << ::std::endl
#  define ACV_DEBUG_COMMAND(COMMAND) COMMAND
#else
#  define ACV_DEBUG_PRINT(ARGS)
#  define ACV_DEBUG_COMMAND(COMMAND)
#endif

#endif // ACV_CONFIG_H
