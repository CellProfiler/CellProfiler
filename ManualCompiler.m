function ManualCompiler %#ok We want to ignore MLint error checking for this line.
% RAY, this is the updated version, Thursday evening.

fid = fopen('CellProfilerManual.tex', 'w');

fwrite(fid,tex_start());

% 1. Cover page (CPCoverPage.ppt was made in powerpoint - would be
% nice if we could extract directly from that, but if not I saved it
% as CPCoverPage.tif)

fwrite(fid,tex_page(tex_center(tex_image('CPCoverPage.png', '1.0\textwidth'))));

% 2. About CellProfiler page (credits) (CPCredits.ppt was made in
% powerpoint - would be nice if we could extract directly from that,
% but if not I saved it as CPCredits.tif)

fwrite(fid,tex_page(tex_center(tex_image('CPCredits.png', '1.0\textwidth'))));

% 3. List of modules - Retrieve the list of files starting with Alg;
% each module is annotated on the line after "Help for the X module:",
% so if you search for the text after the Category:, you will find one
% of five categories: Object Identification, Measurement,
% Pre-processing, File Handling, Other.  I would like these five
% headings to be listed in that order, with the list of available
% algorithms in each category following its heading.  This need not be
% a true table of contents, because page numbers need not be listed.
% The modules are going to be in ABC order anyway, so I think page
% numbers are unnecessary.

filelist = dir('Alg*.m');

fwrite(fid,tex_twocolumn(tex_center(tex_huge('Modules\\[3em]'))));
fwrite(fid,tex_bold(tex_large('Object Identification:\\')));
for i=1:length(filelist),
  if file_in_category(filelist(i).name, 'Object Identification'),
    fwrite(fid,tex_toc_entry(filelist(i).name));
  end
end
fwrite(fid,tex_pagebreak());
fwrite(fid,tex_bold(tex_large('Measurement:\\')));
for i=1:length(filelist),
  if file_in_category(filelist(i).name, 'Measurement'),
    fwrite(fid,tex_toc_entry(filelist(i).name));
  end
end
fwrite(fid,tex_vertical_space('1em'));
fwrite(fid,tex_bold(tex_large('File Handling:\\')));
for i=1:length(filelist),
  if file_in_category(filelist(i).name, 'File Handling'),
    fwrite(fid,tex_toc_entry(filelist(i).name));
  end
end
fwrite(fid,tex_vertical_space('1em'));
fwrite(fid,tex_bold(tex_large('Other:\\')));
for i=1:length(filelist),
  if file_in_category(filelist(i).name, 'Other'),
    fwrite(fid,tex_toc_entry(filelist(i).name));
  end
end
fwrite(fid, tex_onecolumn());


% 3.5 Extract 'help' lines from CellProfiler.m where there is a
% description of CellProfiler and will soon be info about the Example
% Image analysis.

fwrite(fid, tex_page(tex_preformatted(help('CellProfiler.m'))));

% 4. Extract 'help' lines from CPInstallGuide.m, have the title of the
% page be "CPInstallGuide", or "CellProfiler Installation Guide", if
% that's convenient.
fwrite(fid, tex_page(tex_preformatted(help('CPInstallGuide.m'))));

% 5. Screenshot of CellProfiler, with the Data button pushed so the
% Data buttons are revealed.  I have saved it as a CPscreenshot.TIF
% (within ExampleImages), but maybe we could have this automatically
% produced eventually?

fwrite(fid, tex_page(tex_image('CPScreenshot.png', '1.0\textwidth')));

% 6. Extract 'help' lines from Help1.m through Help4.m. Have the title
% of each page be "HelpN".

% 7. Extract 'help' lines from HelpZZZ, where ZZZ is anything else (I
% guess the order is not critical here).

filelist = dir('Help*.m');
for i=1:length(filelist),
  base = basename(filelist(i).name);
  fwrite(fid,tex_page([tex_center(tex_huge(['CellProfiler Help: ' base(5:end) '\\'])) tex_preformatted(help(filelist(i).name))]));
end

% 7.5. Extract 'help' lines from ProgrammingNotes.m

fwrite(fid,tex_page([tex_center(tex_huge('Programming Notes\\')) tex_preformatted(help('ProgrammingNotes.m'))]));

% 8. Open each algorithm (alphabetically) and print its name in large
% bold font at the top of the page (perhaps we should have the
% official "AlgBlaBla.m" name in small font as a subtitle, and the
% "Bla Bla" version of the name, extracted from the "Help for the Bla
% Bla module" line as the title for the page). Extract the lines after
% "Help for ...." and before the license begins, using the Matlab
% 'help' function. Print this below the algorithm name. Extract the
% variables from the algorithm using the same code CP uses, and print
% this below the algorithm description. Open the corresponding tif
% image file (in the ExampleImages folder, these always have the exact
% name as the algorithm), if it exists, and place this at the bottom
% of the page. [somehow deal with it if there is too much stuff to fit
% on one page, though I think at the moment each module should fit on
% one page.]

filelist = dir('Alg*.m');

for i=1:length(filelist),
  base = basename(filelist(i).name);
  base = base(4:end);
  heading = tex_center(tex_huge(['Module: ' base '\\']));
  body = [tex_label(['Alg' base]) tex_preformatted(help(filelist(i).name))];
  im = '';
  if (length(dir(['ExampleImages/' base '.*'])) > 0),
    im = tex_center(tex_image(base, '1.0\textwidth'));
  end
  fwrite(fid,tex_page([heading body im]));
end

% 9. Add page numbers throughout.

% 10. Save as a pdf.

% 11. Doublecheck that all the modules are present, and that all the
% help was included (every module's help should end with "See also").

fwrite(fid, tex_end());
fclose(fid);



function s = tex_start()
s = '\documentclass[letter]{article}\usepackage{graphicx}\setlength{\parindent}{0in}\setlength{\oddsidemargin}{0in}\setlength{\evensidemargin}{0in}\setlength{\topmargin}{0in}\setlength{\textwidth}{6.5in}\setlength{\textheight}{9.0in}\begin{document}\sffamily';


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
endm = strfind(filename, '.m');
s = [filename(4:(endm-1)) '\dotfill \pageref{' filename(1:(endm-1)) '}\\'];


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
