The main files in the Algorithm folder are

1. GENERATE.py, which generates original marker pattern in standard position and stores the output named SAMPLExx.png in the Resource folder

2. EXTRACT.py, which computes the boundary signals, extracts the coordinate system, and computes the angle that rotates the detected pattern to the original pattern in the standard position

3. TRANSFM.py, which applies translation, rotation and resizing to each original marker pattern to generate test sample for classification

4. CLASSFY.py, which classifies the detected pattern and retrieves the corresponding id.

5. FUNCS.py, which defines commonly used functions in the algorithm

6. CFNS.pyx, which is a c wrapper for some of the functions in FUNCS.py to improve performance

7. RUN.py, which runs an animation distinguishing btw two sets of marker patterns, where each set consists of 4 unique marker patterns; the marker patterns are files SAMPLExx.png in the Resource folder


-----------------------------------------------------------------------------------------------------------

Some background for the algorithms in files EXTRACT.py and CLASSFY.py

1. EXTRACT.py

	- the algorithm first computes the boundary signals from the input image, which are implemented by the functions LGN(), Simple() and Complex(), then it extracts the embedded coordinate system, which is implemented by the function CoordSys()
	
	- in each of these functions, the main mechanism is such that at each pixel position of an image is placed a filter and the filter processes the pixel value. The filters typically have two parts, where one part is called on-center that excites the nearby pixel positions and another part is called off-surround that inhibits pixel values at farther positions. The function of an on-center off-surround filter is to contrast enhance the strong signal in the center and inhibit the weak signal in the surround so as to more effectively suppress noise and extract signal. Moreover, if the filters are oriented where filter activity is biased in some orientation, then boundary signals can be computed.


2. CLASSFY.py

	- the algorithm is called ARTMAP in the literature and it is used to learn categories of input patterns, where the input patterns may be arbitrary, though they typically require preprocessing before the algorithm can be applied
	
	- the idea of the algorithm is that given an input pattern, it triggers activation of a category or a compressed code. The code may represent the input pattern or it may not, and this is evaluated by comparing the pattern encoded by the compressed code and the input pattern itself. If the mismatch is large enough, a new category becomes active to encode the input pattern. This cycle of match, mismatch, and learning new categories repeats for each input pattern









------------------------------------------------------------------------------------------------------------

API for each main file


1. GENERATE.py

	-


2. EXTRACT.py

	- 

3. TRANSFM.py

	-

4. CLASSFY.py

	- it us 








