function ManualCompiler %#ok We want to ignore MLint error checking for this line.
%%% The Manual Compiler must be run from the main CellProfiler
%%% directory.

fid = fopen('CellProfilerManual.tex', 'w');
fwrite(fid,tex_start());

% 1. Cover page & About CellProfiler page (credits). These pages were
% made in powerpoint - would be nice if we could extract directly from
% that, but in the meantime, they are saved png image files.
fwrite(fid,tex_page(tex_center(tex_image('CPCoverPage.png', '1.0\textwidth'))));
fwrite(fid,tex_page(tex_center(tex_image('CPCredits.png', '1.0\textwidth'))));

% 2. Table of contents - Retrieve the list of modules.
% each module is annotated on the line after "Help for the X module:",
% so searching for the text after the Category:, will find one
% of five categories: Object Identification, Measurement,
% Pre-processing, File Handling, Other.  The five
% headings are listed in that order, with the list of available
% algorithms in each category following its heading.
path(fullfile(pwd,'Modules'), path)
filelist = dir('Modules/*.m');
fwrite(fid,tex_twocolumn(tex_center(tex_huge('Modules\\[3em]'))));
fwrite(fid,tex_bold(tex_large('Object Identification:\\')));
for i=1:length(filelist),
  if file_in_category(['Modules/' filelist(i).name], 'Object Identification'),
    fwrite(fid,tex_toc_entry(filelist(i).name));
  end
end
fwrite(fid,tex_vertical_space('1em'));
fwrite(fid,tex_bold(tex_large('Measurement:\\')));
for i=1:length(filelist),
  if file_in_category(['Modules/' filelist(i).name], 'Measurement'),
    fwrite(fid,tex_toc_entry(filelist(i).name));
  end
end
fwrite(fid,tex_pagebreak());
fwrite(fid,tex_bold(tex_large('Pre-processing:\\')));
for i=1:length(filelist),
  if file_in_category(['Modules/' filelist(i).name], 'Pre-processing'),
    fwrite(fid,tex_toc_entry(filelist(i).name));
  end
end
fwrite(fid,tex_vertical_space('1em'));
fwrite(fid,tex_bold(tex_large('File Handling:\\')));
for i=1:length(filelist),
  if file_in_category(['Modules/' filelist(i).name], 'File Handling'),
    fwrite(fid,tex_toc_entry(filelist(i).name));
  end
end
fwrite(fid,tex_vertical_space('1em'));
fwrite(fid,tex_bold(tex_large('Other:\\')));
for i=1:length(filelist),
  if file_in_category(['Modules/' filelist(i).name], 'Other'),
    fwrite(fid,tex_toc_entry(filelist(i).name));
  end
end
fwrite(fid, tex_onecolumn());

% 3 Extract 'help' lines from CellProfiler.m where there is a
% description of CellProfiler and the Example image analysis.
fwrite(fid, tex_page(tex_preformatted(help('CellProfiler.m'))));

% 4. Extract 'help' lines from CPInstallGuide.m, have the title of the
% page be "CPInstallGuide", or "CellProfiler Installation Guide", if
% that's convenient.
fwrite(fid, tex_page(tex_preformatted(help('HelpCPInstallGuide.m'))));

% 5. Screenshot of CellProfiler (within ExampleImages), but maybe we
% could have this automatically produced eventually?
fwrite(fid, tex_page(tex_image('CPScreenshot.png', '1.0\textwidth')));

% 6. Extract 'help' lines from anything in the help folder starting
% with 'Help' (the order is not critical here).
path(fullfile(pwd,'Help'), path);
filelist = dir('Help/Help*.m');
for i=1:length(filelist),
  base = basename(filelist(i).name);
  if (strcmp(base, 'HelpCPInstallGuide') == 1),
      continue;
  end
  fwrite(fid,tex_page([tex_center(tex_huge(['CellProfiler Help: ' base(5:end) '\\'])) tex_preformatted(help(filelist(i).name))]));
end

% 7. Open each module (alphabetically) and print its name in large
% bold font at the top of the page. Extract the lines after "Help for
% ...." and before the license begins, using the Matlab 'help'
% function. Print this below the module name. Open the corresponding
% image file (in the ExampleImages folder, these always have the exact
% name as the algorithm), if it exists, and place this at the bottom
% of the page.
filelist = dir('Modules/*.m');
for i=1:length(filelist),
  if file_in_category(['Modules/' filelist(i).name], 'Testing'),
    continue;
  end
  base = basename(filelist(i).name);
  heading = tex_center(tex_huge(['Module: ' base '\\']));
  body = [tex_label(['Module:' base]) tex_preformatted(help(filelist(i).name))];
  im = '';
  if (length(dir(['ExampleImages/' base '.*'])) > 0),
    im = tex_center(tex_image(base, '1.0\textwidth'));
  end
  fwrite(fid,tex_page([heading body im]));
end
fwrite(fid, tex_end());
fclose(fid);

%%% SUBFUNCTIONS %%%
function s = tex_start()
s = '\documentclass[letterpaper]{article}\usepackage{graphicx}\setlength{\parindent}{0in}\setlength{\oddsidemargin}{0in}\setlength{\evensidemargin}{0in}\setlength{\topmargin}{0in}\setlength{\headheight}{0in}\setlength{\headsep}{0in}\setlength{\textwidth}{6.5in}\setlength{\textheight}{9.0in}\begin{document}\sffamily';


function s = tex_end()
s = '\end{document}';

function sout = tex_page(sin)
sout = [sin '\newpage'];

function sout = tex_center(sin)
sout = ['\begin{center}' sin '\end{center}'];

function sout = tex_image(sin, width)
sout = ['\includegraphics[width=' width ']{ExampleImages/' sin '}'];

function sout = tex_huge(sin)
sout = ['{\huge ' sin '}'];

function sout = tex_large(sin)
sout = ['{\large ' sin '}'];

function sout = tex_bold(sin)
sout = ['{\bfseries ' sin '}'];

function s = tex_toc_entry(filename)
b = basename(filename);
s = [b '\dotfill \pageref{Module:' b '}\\'];


function c = file_in_category(filename, category)
h = help(filename);
if strfind(h, ['Category: ' category]),
  c = 1;
else,
  c = 0;
end

function sout = tex_twocolumn(header)
sout = ['\twocolumn[' header ']'];

function s = tex_onecolumn()
s = '\onecolumn';

function s = tex_pagebreak()
s = '\newpage ';

function sout = tex_vertical_space(sin)
sout = ['\vspace{' sin '}\\'];

function sout = tex_preformatted(sin)
sout = ['\begin{verbatim}' sin '\end{verbatim}'];

function b=basename(filename)
idx = strfind(filename, '.m');
b = filename(1:(idx-1));
if (strfind(b, '_')),
  loc = strfind(b, '_');
  b = [b(1:loc-1)  b(loc+1:end)];
end

function sout = tex_label(sin)
sout = ['\label{' sin '}'];

Result = 'Compilation is complete'