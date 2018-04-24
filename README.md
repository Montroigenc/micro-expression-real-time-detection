# micro-expression-real-time-detection

install https://cmake.org/download/

Download http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Download http://www.boost.org/ and execute in the command prompt in the directory where you have unzipped boost:
##########################################################
bootstrap.bat #First run the bootstrap.bat file supplied with boost-python

#Once it finished invoke the install process of boost-python like this:
b2 install #This can take a while, go get a coffee

#Once this finishes, build the python modules like this
b2 -a --with-python address-model=64 toolset=msvc runtime-link=static #Again, this takes a while, reward yourself and get another coffee.
##########################################################

Download http://dlib.net/ and execute in the command prompt in the directory where you have unzipped dlib:
##########################################################
# Set two flags so that the CMake compiler knows where to find the boost-python libraries
set BOOST_ROOT=C:\boost #Make sure to set this to the path you extracted boost-python to!
set BOOST_LIBRARYDIR=C:\boost\stage\lib #Same as above

# Create and navigate into a directory to build into
mkdir build
cd build

# Build the dlib tools
cmake ..

#Navigate up one level and run the python setup program
cd ..
python setup.py install #This takes some time as well. GO GET ANOTHER COFFEE TIGER!
##########################################################

This code has been generated using http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/ as a base.
