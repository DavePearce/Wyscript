// This file is part of the WyScript Compiler (wysc).
//
// The WyScript Compiler is free software; you can redistribute
// it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// The WyScript Compiler is distributed in the hope that it
// will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public
// License along with the WyScript Compiler. If not, see
// <http://www.gnu.org/licenses/>
//
// Copyright 2013, David James Pearce.

package wyscript.util;

public interface Attribute {

  public static class Source implements Attribute {

    public final int start;
    public final int end;

    public Source(int start, int end) {
      this.start = start;
      this.end = end;
    }

    public String toString() {
      return "@" + start + ":" + end;
    }
  }
  
  public static class Type implements Attribute {
	  public final wyscript.lang.Type type;
	  
	  public Type(wyscript.lang.Type type) {
		  this.type = type;
	  }
  }
}
