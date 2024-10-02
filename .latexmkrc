#$lualatex = 'lualatex -shell-escape -interaction=batchmode -file-line-error %O %S';
$pdflatex = 'pdflatex -shell-escape -interaction=batchmode -file-line-error %O %S';
   $max_repeat = 5;
   $pdf_mode = 1;
   @default_files = ('main.tex');

# Set up output directory
$out_dir = 'build';

# Ensure the build directory exists
use File::Path qw(make_path);
make_path($out_dir);
