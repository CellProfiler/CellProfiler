function [objectC,IA,IB] = objdiff(objectA,objectB,varargin)
% OBJDIFF  compares two objects & returns an object of the same type with just the different fields/values.
%
%   OBJDIFF (unlike Matlab's SETDIFF or SETXOR) also compares structs, GUI
%   handles, ActiveX, Matlab & Java objects, in addition to arrays & cells.
%   OBJDIFF also allows comparison of numeric cell arrays, unlike SETDIFF/
%   SETXOR. It also accepts everything that SETDIFF/SETXOR accept.
%
%   Syntax: [objectC,IA,IB] = objdiff (objectA, objectB, options, ...)
%
%   Inputs:
%     objectA - first object to compare
%     objectB - second object to compare. Field order in opaque objects does not matter.
%               Note: If objectB is not supplied, then objectA(1) is compared to objectA(2)
%     options - optional flags as follows:
%       'rows' - see documentation for <a href="matlab:doc setxor">SETXOR</a>
%       'dontIgnoreJava' - show different instances of the same java class (default=ignore them)
%
%   Outputs:
%     objectC - object containing only the different (or new) fields, in a {old, new} pattern
%     IA,IB - index vector into objectA,objectB such that objectC = [objectA(IA),objectB(IB)] (see SETXOR)
%
%   Examples:
%     >> objectA = struct('a',3, 'b',5, 'd',9);
%     >> objectB = struct('a','ert', 'c',struct('t',pi), 'd',9);
%     >> objectC = objdiff(objectA, objectB)  % a=different, b=new in objectA, c=new in objectB, d=same
%     objectC = 
%         a: {[3]  'ert'}
%         b: {[5]  {}}
%         c: {{}  [1x1 struct]}
%
%     >> objectC = objdiff(java.awt.Color.red, java.awt.Color.blue)
%     objectC = 
%         Blue: {[0]  [255]}
%          RGB: {[-65536]  [-16776961]}
%          Red: {[255]  [0]}
%
%     >> objectC = objdiff(0,gcf)  % 0 is the root handle
%     objectC = 
%           children: {[2x1 struct]  []}
%             handle: {[0]  [1]}
%         properties: {[1x1 struct]  [1x1 struct]}
%               type: {'root'  'figure'}
%
%     >> [objectC,IA,IB] = objdiff({2,3,4,7}, {2,4,5})
%     objectC =
%          3     5     7
%     IA =
%          2     4
%     IB =
%          3
%
%   Bugs and suggestions:
%     Please send to Yair Altman (altmany at gmail dot com)
%
%   Change log:
%     2007-07-27: Fixed handling of identical objects per D. Gamble
%     2007-03-23: First version posted on <a href="http://www.mathworks.com/matlabcentral/fileexchange/loadAuthor.do?objectType=author&mfx=1&objectId=1096533#">MathWorks File Exchange</a>
%
%   See also:
%     SETDIFF, SETXOR, ISSTRUCT, ISJAVA, ISHGHANDLE, ISOBJECT, ISCELL

% Programmed by Yair M. Altman: altmany(at)gmail.com
% $Revision$  $Date$

  % Process input args
  if (nargin<1) %|| ~isstruct(objectA) || ~isstruct(objectB)
      help objdiff
      error('YMA:OBJDIFF:NotEnoughInputs', 'Not enough input arguments');
  elseif (nargin<2) || (nargin==2 && ~strcmp(class(objectA),class(objectB)))
      if numel(objectA) < 2
          error('YMA:OBJDIFF:NotEnoughInputs', 'Not enough input arguments');
      elseif numel(objectA) > 2
          warning('YMA:OBJDIFF:TooManyInputs', 'Too many elements in objectA - only comparing first 2');
      end
      objectB = objectA(2);
      objectA = objectA(1);
      varargin = {objectB, varargin{:}};
  elseif ~strcmp(class(objectA),class(objectB))
      error('YMA:OBJDIFF:DissimilarObjects', 'Input objects must be of the same type');
  end

  % Process optional options
  ignoreJavaObjectsFlag = true;
  if ~isempty(varargin)
      ignoreJavaIdx = strmatch('dontignorejava',lower(varargin{:}));
      if ~isempty(ignoreJavaIdx)
          ignoreJavaObjectsFlag = false;
          varargin(ignoreJavaIdx) = [];
      end
  end

  % TODO: check for array of java/struct objs

  % Convert opaque objects to structs with the relevant property fields
  if ishghandle(objectA)
      objectA = handle2struct(objectA);
      objectB = handle2struct(objectB);
  else%if isjava(object(A))
      try
          % This should work for any opaque object: Java, ActiveX & Matlab
          objectA = get(objectA);
          objectB = get(objectB);
      catch
          % never mind - try to process as-is
      end
  end

  % Enable comparison of numeric cell arrays
  objectA = decell(objectA);
  objectB = decell(objectB);

  % Process based on object type
  if isstruct(objectA)
      % Structs - loop over all fields
      [objectC, IA, IB] = compareStructs(objectA, objectB, ignoreJavaObjectsFlag);
  else
      % Cells and arrays - process with the regular setdiff function
      [objectC, IA, IB] = setxor(objectA, objectB, varargin{:});
  end


%% Compare two structs
function [objectC,IA,IB] = compareStructs(objectA,objectB,ignoreJavaObjectsFlag)
  % Ensure singleton objects are compared
  objectA = getSingleton(objectA);
  objectB = getSingleton(objectB);
  objectC = struct();

  % Get all the fieldnames
  fieldsA = fieldnames(objectA);
  fieldsB = fieldnames(objectB);
  allFields = union(fieldsA, fieldsB);

  % Loop over all fields and compare the objects
  for fieldIdx = 1 : length(allFields)
      fieldName = allFields{fieldIdx};
      if ~isfield(objectA,fieldName)
          objectC.(fieldName) = {{}, objectB.(fieldName)};
      elseif ~isfield(objectB,fieldName)
          objectC.(fieldName) = {objectA.(fieldName), {}};
      elseif ~isequalwithequalnans(objectA.(fieldName), objectB.(fieldName))
          if ignoreJavaObjectsFlag && isjava(objectA.(fieldName)) && isjava(objectB.(fieldName)) && ...
                  isequalwithequalnans(objectA.(fieldName).getClass, objectB.(fieldName).getClass)
              continue;
          end
          objectC.(fieldName) = {objectA.(fieldName), objectB.(fieldName)};
      end
  end

  % Check for empty diff struct (identical input objects)
  if isempty(fieldnames(objectC))
      objectC = struct([]);
  end

  % no use for IA,IB...
  IA = [];
  IB = [];


%% De-cell-ize a numeric cell-array
function obj = decell(obj)
  if iscell(obj) && ~iscellstr(obj)
      obj = cell2mat(obj);
  end


%% Ensure singleton object
function obj = getSingleton(obj)
  if numel(obj) > 1
      warning('YMA:OBJDIFF:TooManyElements', 'Too many elements in %s - only comparing the first', inputname(1));
      obj = obj(1);
  end
