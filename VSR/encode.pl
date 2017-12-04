use strict;

my $num_frames = 5;
for (my $start = 0; $start < 10; $start = $start + 1){
	my $encode = "ffmpeg -i foreman_cif.y4m -s 10x10 -frames:v ${num_frames} -ss ${start} out//s${start}_f%d.bmp";
	system($encode);
}
