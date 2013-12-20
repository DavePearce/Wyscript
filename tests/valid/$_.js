
//FLOAT CLASS/METHODS
var $_Float_$ = function(i) {
	if (typeof i.num !== 'undefined') this.num = i.num;
	else this.num = i;
	this.type = 'real';
};

$_Float_$.prototype.add = function(other) {
	return new $_Float_$(this.num + other.num);
};
		
$_Float_$.prototype.sub = function(other) {
	return new $_Float_$(this.num - other.num);
};

$_Float_$.prototype.mul = function(other) {
	return new $_Float_$(this.num * other.num);
};

$_Float_$.prototype.div = function(other) {
	return new $_Float_$(this.num / other.num);
};

$_Float_$.prototype.rem = function(other) {
	return new $_Float_$(this.num % other.num);
};
		
$_Float_$.prototype.cast = function(type) {
    if (type === 'real')
      return new $_Float_$(this.num);
	
    else return new $_Integer_$(this.num);
};

$_Float_$.prototype.toString = function() {
    var tmp = this.num.toString();
    var abs = Math.abs(this.num);
    if (abs < 0.001 || abs >= 10000000)
	tmp = this.num.toExponential().replace('e', 'E');
	if (tmp.indexOf('.') === -1) {
		if (tmp.indexOf('E') !== -1)
		    tmp = tmp.replace('E', '.0E');
		else tmp += '.0';
	}
	return tmp;
};

//INTEGER CLASS/METHODS
var $_Integer_$ = function(i) {
	this.type = 'int';
	if (typeof i.num !== 'undefined') this.num = ~~(i.num);
	else this.num = ~~i;
};

$_Integer_$.prototype.add = function(other) {
	if (other instanceof $_Integer_$)
		return new $_Integer_$(this.num + other.num);
	
	else return new $_Float_$(this.num + other.num);
};

$_Integer_$.prototype.sub = function(other) {
	if (other instanceof $_Integer_$)
		return new $_Integer_$(this.num - other.num);
		
	else return new $_Float_$(this.num - other.num);
};

$_Integer_$.prototype.mul = function(other) {
	if (other instanceof $_Integer_$)
		return new $_Integer_$(this.num * other.num);
		
	else return new $_Float_$(this.num * other.num);
};

$_Integer_$.prototype.div = function(other) {
	var tmp = this.num / other.num;
	if (other instanceof $_Integer_$)
		return new $_Integer_$(~~tmp);
		
	else return new $_Float_$(this.num / other.num);
};

$_Integer_$.prototype.rem = function(other) {	
	if (other instanceof $_Integer_$)
		return new $_Integer_$(this.num % other.num);
		
	else return new $_Float_$(this.num % other.num);
};
		
$_Integer_$.prototype.cast = function(type) {
	if (type === 'int')
	  return new $_Integer_$(this.num);
	  
	else return new $_Float_$(this.num);
};
$_Integer_$.prototype.toString = function() {
	return this.num.toFixed();
};

//BINARY METHODS
function $_range_$(lower, upper) {
  var low = lower;
  var up = upper;
  if (typeof low.num !== 'undefined') low = lower.num;
  if (typeof up.num !== 'undefined') up = upper.num;
  var result = [];
  var count = 0;
  for (var tmp = low; tmp < up; tmp++) {
    result[count++] = new $_Integer_$(tmp);
  }
  return result;
}

function $_append_$(left, right) {
  if (left instanceof String || typeof left === 'string')
    return left.concat(right.toString());
  else return left.append(right);
}

function $_equals_$(lhs, rhs, isEqual) {
  var left = lhs;
  if (typeof left.num !== 'undefined') left = left.num;
  else if (left instanceof String) left = left.valueOf();
  else if (left instanceof $_List_$) {
    if (isEqual) return left.equals(rhs);
    else return !(left.equals(rhs));
  }
  var right = rhs;
  if (typeof right.num !== 'undefined') right = right.num;
  else if (right instanceof String) right = right.valueOf;
  if (isEqual) return left === right;
  else return left !== right;
}

function $_lt_$(lhs, rhs, isEqual) {
  var left = lhs;
  if (typeof left.num !== 'undefined') left = left.num;
  var right = rhs;
  if (typeof right.num !== 'undefined') right = right.num;
  if (isEqual) return (left <= right);
  else return (left < right);
}

function $_gt_$(lhs, rhs, isEqual) {
  var left = lhs;
  if (typeof left.num !== 'undefined') left = left.num;
  var right = rhs;
  if (typeof right.num !== 'undefined') right = right.num;
  if (isEqual) return (left >= right);
  else return (left > right);
}

//RECORD CLASS/METHODS
var $_Record_$ = function(listNames, listValues, type) {
  this.names = listNames;
  this.values = listValues;
  this.type = type;
};

$_Record_$.prototype.getValue = function(name) {
  var index = this.names.indexOf(name);
  if (index === -1 || index >= this.values.length)
    return null;
  else {
    var elem = this.values[index];
    return elem;
  }
};

$_Record_$.prototype.hasKey = function(name) {
  return (this.names.indexOf(name) !== -1);
};

$_Record_$.prototype.setValue = function(name, key) {
  var index = this.names.indexOf(name);
  if (index === -1 || index >= this.values.length)
    return;
  else this.values[index] = key;
};

$_Record_$.prototype.cast = function(name, fieldList, newType) {
  var result = this.clone();
  if (fieldList.length > 0) {
    var index = this.names.indexOf(fieldList[0]);
    result.values[index] = this.values[index].cast(name, fieldList.slice(1));
  }
  else {
    var index = this.names.indexOf(name);
    result.values[index] = this.values[index].cast();
  }
  result.type = newType;
  return result;
};

$_Record_$.prototype.clone = function() {
  var cnames = [];
  var cvalues = [];
  for (var i = 0; i < this.names.length; i++) {
    cnames[i] = this.names[i];
    var elem = this.values[i];
    if (elem instanceof $_List_$ || elem instanceof $_Record_$)
      elem = elem.clone();
    cvalues[i] = elem;
  }
  return new $_Record_$(cnames, cvalues, this.type);
};

$_Record_$.prototype.toString = function() {
  var str = '{';
  var tmpNames = [];
  for (var i = 0; i < this.names.length; i++) {
    tmpNames[i] = this.names[i];
    tmpNames.sort();
  }
  var first = true;
  for (i = 0; i < this.names.length; i++) {
    if (!first)
      str += ',';
    first = false;
    str += tmpNames[i];
    str += ':';
    str += this.values[this.names.indexOf(tmpNames[i])].toString();
  }
  str += '}';
  return str;
};

//LIST FUNCTION
var $_List_$ = function(list, type) {
  this.list = list;
  this.type = type;
};

$_List_$.prototype.getValue = function(index) {
  var idx = index;
  if (typeof index.num !== 'undefined') idx = index.num;
  return this.list[idx];
};

$_List_$.prototype.setValue = function(index, value) {
  var idx = index;
  if (typeof index.num !== 'undefined') idx = index.num;
  this.list[idx] = value;
};

$_List_$.prototype.length = function() {
  return new $_Integer_$(this.list.length);
};

$_List_$.prototype.append = function(other) {
  var result = [];
  var count = 0;
  for (var i = 0; i < this.list.length; i++)
    result[count++] = this.list[i];
  for (var i = 0; i < other.list.length; i++)
    result[count++] = other.list[i];
  return new $_List_$(result);
};

$_List_$.prototype.cast = function(name, fieldList, newType) {
  var tmp = [];
  for (var i = 0; i < this.list.length; i++)
    tmp[i] = this.list[i].cast(name, fieldList);
  return new $_List_$(tmp, newType);
};

$_List_$.prototype.clone = function() {
  var clist = [];
  for (var i = 0; i < this.list.length; i++) {
    var elem = this.list[i];
    if (elem instanceof $_List_$ || elem instanceof $_Record_$)
      elem = elem.clone();
    clist[i] = elem;
  }
  return new $_List_$(clist, this.type);
};

$_List_$.prototype.toString = function() {
  var str = '[';
  var first = true;
  for (var i = 0; i < this.list.length; i++) {
    if (!first)
      str += ', ';
    
    first = false;
    str += this.list[i];
  }
  str += ']';
  return str;
};

$_List_$.prototype.equals = function(other) {
  if (!other instanceof $_List_$ || typeof other === 'undefined')
    return false;
  if (this.length().num !== other.length().num) return false;
  for (var i = 0; i < this.list.length; i++) {
    if (!($_equals_$(this.list[i], other.list[i], true)))
      return false;
  }
  return true;
};

function $_stringIndexReplace_$(str, index, c) {
  var num = index;
  if (typeof index.num !== 'undefined')
    num = index.num;
  var tmp = str.split('');
  tmp[num] = c;
  return tmp.join('');
}

function $_indexOf_$(obj, index) {
  if (obj instanceof String || typeof obj === 'string')
    return obj.charAt(index);
  else return obj.getValue(index);
}

function $_length_$(obj) {
  if (obj instanceof String || typeof obj === 'string')
    return new $_Integer_$(obj.length);
  else return obj.length();
}

function $_print_$(obj) {
  if (obj === null)
    sysout.println('null');
  else sysout.println(obj.toString());
}

function $_is_$(obj, type) {
  if (type === 'void')
      return true;
      
  else if (type === 'null')
      return (obj === null);
      
  else if (type === 'bool')
      return (obj instanceof Boolean || typeof obj === 'boolean');
      
  else if (type === 'string')
      return (obj instanceof String || typeof obj === 'string');
      
  else if (type === 'char') {
      if (!(obj instanceof String || typeof obj === 'string'))
	  return false;
      else return (obj.length === 1);
  }
  
  else if (type === 'real')
      return (obj instanceof $_Float_$);
      
  else if (type === 'int')
      return (obj instanceof $_Integer_$);
      
  else if (obj instanceof $_List_$ || obj instanceof $_Record_$)
      return (obj.type == type);
      
  else return false; //Unknown type
  
}