
form choose a sound file
    text wave_file sx307
    text pitch_name sx307
    text out_dir ../pitch
    text top 300
    text bot 50
endform

sound = Read from file: wave_file$ 
plusObject: sound
selectObject: sound
out_file$ = out_dir$ + "/" + pitch_name$ + ".pitch"

bot = number(bot$)
top = number(top$)

To Pitch: 0.005, bot, top
no_of_frames = Get number of frames


writeFileLine: out_file$, " time,pitch"
for frame from 1 to no_of_frames
    time = Get time from frame number: frame
    pitch = Get value in frame: frame, "Hertz"
    appendFileLine: out_file$, "'time','pitch'"
endfor
