project('neuralnetwok', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++17'])

# 1. Find the SFML libraries on your system
# Since you are running in "Host" mode, this will find the apt-get installed files
sfml_graphics = dependency('sfml-graphics')
sfml_window   = dependency('sfml-window')
sfml_system   = dependency('sfml-system')

# 2. Build the executable
executable('neuralnetwok',
           'src/main.cpp',
           install : true,
           # CRITICAL: You must list the libraries here so the linker uses them
           dependencies : [sfml_graphics, sfml_window, sfml_system]
)

executable('draw',
           'src/draw.cpp',
           install : true,
           dependencies : [sfml_graphics, sfml_window, sfml_system]
)