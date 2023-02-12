# CatPatch
CatPatch is an application to cut images into small patches. It uses a graphical interface created in Python with the tkinter and matplotlib modules.

## Usage
1. A button to load the image to cut
2. An area to enter the dimensions of the patches to cut (width x height) with a "Cut" button that displays a red grid on the image to show the borders of the patches.
3. After clicking "Cut", you can click on patches in the cut image to gray them out. If you click on a grayed out patch again, it will return to normal.
4. A "Save" button to save the unshaded image patches to a user-specified folder. The grayed out patches will be saved in a folder called "NL" (No Label). The images will be saved with the original file name followed by an underscore and a unique number, in PNG format.

