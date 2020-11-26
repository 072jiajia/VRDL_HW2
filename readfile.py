import h5py
import numpy as np


class BBox:
    ''' Bounding Box '''

    def __init__(self):
        self.label = ""     # Digit
        self.left = 0
        self.top = 0
        self.width = 0
        self.height = 0


class DigitStruct:
    ''' filename and BBox list '''

    def __init__(self):
        self.name = None    # Image file name
        self.bboxList = None  # List of BBox structs


def readDigitStructGroup(dsFile):
    dsGroup = dsFile["digitStruct"]
    return dsGroup


def readString(strRef, dsFile):
    ''' Reads a string from the file using its reference '''
    strObj = dsFile[strRef]
    str = ''.join(chr(i) for i in strObj)
    return str


def readInt(intArray, dsFile):
    ''' Reads an integer value from the file '''
    intRef = intArray[0]
    isReference = isinstance(intRef, h5py.Reference)
    intVal = 0
    if isReference:
        intObj = dsFile[intRef]
        intVal = int(intObj[0])
    else:  # Assuming value type
        intVal = int(intRef)
    return intVal


def yieldNextInt(intDataset, dsFile):
    for intData in intDataset:
        intVal = readInt(intData, dsFile)
        yield intVal


def yieldNextBBox(bboxDataset, dsFile):
    for bboxArray in bboxDataset:
        bboxGroupRef = bboxArray[0]
        bboxGroup = dsFile[bboxGroupRef]
        labelDataset = bboxGroup["label"]
        leftDataset = bboxGroup["left"]
        topDataset = bboxGroup["top"]
        widthDataset = bboxGroup["width"]
        heightDataset = bboxGroup["height"]

        left = yieldNextInt(leftDataset, dsFile)
        top = yieldNextInt(topDataset, dsFile)
        width = yieldNextInt(widthDataset, dsFile)
        height = yieldNextInt(heightDataset, dsFile)

        bboxList = []

        for label in yieldNextInt(labelDataset, dsFile):
            bbox = BBox()
            bbox.label = label
            bbox.left = next(left)
            bbox.top = next(top)
            bbox.width = next(width)
            bbox.height = next(height)
            bboxList.append(bbox)

        yield bboxList


def yieldNextFileName(nameDataset, dsFile):
    for nameArray in nameDataset:
        nameRef = nameArray[0]
        name = readString(nameRef, dsFile)
        yield name


def yieldNextDigitStruct(dsFileName):
    dsFile = h5py.File(dsFileName, 'r')
    dsGroup = readDigitStructGroup(dsFile)
    nameDataset = dsGroup["name"]
    bboxDataset = dsGroup["bbox"]

    bboxListIter = yieldNextBBox(bboxDataset, dsFile)
    for name in yieldNextFileName(nameDataset, dsFile):
        bboxList = next(bboxListIter)
        obj = DigitStruct()
        obj.name = name
        obj.bboxList = bboxList
        yield obj


def main():
    ''' Generate annotations of images
    each xxx.png has a file of annotations named xxx.npy
    '''
    dsFileName = 'data/train/digitStruct.mat'
    i = 0
    for dsObj in yieldNextDigitStruct(dsFileName):
        print(i, end='\r')
        i += 1
        array = []
        for bbox in dsObj.bboxList:
            array.append([bbox.left, bbox.top, bbox.left +
                          bbox.width, bbox.top + bbox.height, bbox.label % 10])

        array = np.array(array)
        filename = 'data/train/' + dsObj.name[:-4] + '.npy'
        np.save(filename, array)


if __name__ == "__main__":
    main()
