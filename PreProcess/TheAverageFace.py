
#The idea and the algorithm has been taken from the link below
#Resoruce: https://www.learnopencv.com/average-face-opencv-c-python-tutorial/
#(https://github.com/spmallick/learnopencv/blob/master/FaceAverage/faceAverage.py)

#Because of huge datset and limited memory when I was running it on my laptop, I had to do it in steps

#Run this: python3 TheAverageFace.py -Test False -Input ./UNBC -Output ./UNBC_Warped -Frames False
import os
import cv2
import numpy as np
import math
import sys
import functools
import gc
import argparse
from os.path import isfile, join


fps = 29.97
frames = False
#The standard width and height for all output images that I scale the dataset to
# If width and height needs to be changed, change the #of triangles in func calculateDelaunayTriangles
w = 320
h = 320 
idx = 0
#Change this depending on the total number of frames
num_frames = 0

#The code is implemented as such it works in three steps:

#Step 1      Simply reads all the landmark points in one big numpy file called allPoints.npy

#Step 2  Finds the AverageFace landmark points from all the normalized images and saves it in a file called AverageLandmarks.npy

#Step 3  Normalises each image, warp them to average face and then save them, including the transformed landmark points


#Note I have been garbage collecting (gc) because I kept running out of memory on my pc


########################################################## Functions for step 1#######################################################

#for counting number of frames
def countFrames(path) :
    global num_frames
    #List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        if filePath.endswith(".txt"):
            num_frames += 1
    return None


# Read points from text files in directory
# Reads for one sequnce video of one person
def readPoints(path) :
    global num_frames
    # Create an array of array of points.
    pointsArray = []
    
    #List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        
        if filePath.endswith(".txt"):
            num_frames += 1
            
            #Create an array of points.
            points = []
            
            # Read points from filePath
            with open(os.path.join(path, filePath)) as file :
                for line in file :
                    x, y = line.split()
                    points.append((float(x), float(y)))
                file.close()
            # Store array of points
            pointsArray.append(points)
            
    return pointsArray



#############################################Common functions for step 2 and step 3####################################################

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are putting dummy third one.

def similarityTransform(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)  
  
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
    
    inPts.append([np.int(xin), np.int(yin)])

    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
    
    outPts.append([np.int(xout), np.int(yout)])
    
    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
    
    return tform[0]


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True




# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)
   
    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))
 
    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array
    delaunayTri = []
    
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        #change here if getting white patches or if the width and height needs to be changes
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    #if(abs(pt[j][0] - points[k][0]) < (320/600) and abs(pt[j][1] - points[k][1]) < (320/600)):
                    if(abs(pt[j][0] - points[k][0]) < 0.1 and abs(pt[j][1] - points[k][1]) < 0.1):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
      
    #correct len of DelaunayTri = 138  
    #delaunayTri = [(36, 18, 37), (18, 36, 17), (2, 73, 1), (73, 2, 3), (0, 73, 66), (73, 0, 1), (60, 51, 61), (51, 60, 50), (1, 0, 36), (7, 57, 8), (57, 7, 58), (73, 3, 4), (2, 1, 31), (29, 31, 40), (31, 29, 30), (72, 4, 5), (4, 72, 73), (3, 2, 31), (52, 34, 35), (34, 52, 51), (72, 5, 6), (4, 3, 48), (71, 72, 6), (5, 4, 48), (35, 54, 53), (54, 35, 14), (6, 5, 59), (38, 40, 37), (40, 38, 39), (71, 6, 7), (7, 6, 58), (38, 21, 39), (21, 38, 20), (71, 7, 8), (9, 70, 71), (70, 9, 10), (9, 71, 8), (9, 8, 56), (70, 10, 11), (10, 9, 55), (30, 35, 34), (35, 30, 29), (70, 11, 12), (11, 10, 54), (14, 35, 46), (69, 70, 13), (12, 11, 54), (35, 29, 47), (13, 70, 12), (13, 12, 54), (1, 36, 41), (69, 13, 14), (14, 13, 54), (37, 18, 19), (68, 69, 16), (14, 15, 69), (15, 14, 46), (37, 20, 38), (20, 37, 19), (16, 69, 15), (16, 15, 45), (31, 1, 41), (0, 66, 17), (17, 36, 0), (67, 23, 22), (23, 67, 24), (18, 17, 66), (18, 67, 19), (67, 18, 66), (67, 22, 21), (67, 20, 19), (20, 67, 21), (23, 43, 22), (43, 23, 24), (21, 22, 27), (42, 27, 22), (27, 42, 28), (42, 22, 43), (29, 42, 47), (42, 29, 28), (43, 24, 44), (44, 46, 47), (46, 44, 45), (24, 67, 25), (26, 45, 25), (45, 26, 16), (24, 25, 44), (25, 68, 26), (68, 25, 67), (16, 26, 68), (31, 41, 40), (21, 27, 39), (48, 31, 49), (31, 48, 3), (27, 28, 39), (28, 29, 39), (10, 55, 54), (49, 32, 50), (32, 49, 31), (58, 61, 64), (61, 58, 65), (31, 30, 32), (60, 61, 65), (32, 30, 33), (52, 64, 51), (64, 52, 62), (32, 33, 50), (33, 30, 34), (33, 34, 51), (15, 46, 45), (44, 47, 43), (37, 40, 41), (36, 37, 41), (29, 40, 39), (44, 25, 45), (42, 43, 47), (46, 35, 47), (5, 48, 59), (48, 49, 59), (50, 33, 51), (49, 50, 60), (56, 8, 57), (63, 53, 55), (53, 63, 62), (52, 35, 53), (54, 55, 53), (52, 53, 62), (55, 9, 56), (64, 63, 56), (63, 64, 62), (55, 56, 63), (49, 60, 59), (56, 57, 64), (57, 58, 64), (58, 6, 59), (58, 59, 65), (59, 60, 65), (61, 51, 64)] 
    return delaunayTri



def constrainPoint(p, w, h) :
    p =  ( min( max( p[0], 0 ) , w - 1 ) , min( max( p[1], 0 ) , h - 1 ) )
    return p





# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst



# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :


    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect




#called for every frame, just normalising based on the postion of the eye
def coordinate_transform(point,image):
    global w,h

    # Eye corners
    eyecornerDst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ]
     
    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])
    

    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
        
    # Corners of the eye in input image
    eyecornerSrc  = [ point[36], point[45] ] 

    # Compute similarity transform
    tform = similarityTransform(eyecornerSrc, eyecornerDst)
    # Apply similarity transformation
    img = cv2.warpAffine(image, tform, (w,h))

    # Apply similarity transform on points
    points2 = np.reshape(np.array(point), (66,1,2))
        
    point = cv2.transform(points2, tform)
        
    point = np.float32(np.reshape(point, (66, 2)))
        
    # Append boundary points. Will be used in Delaunay Triangulation
    point = np.append(point, boundaryPts, axis=0)
        
    # Calculate location of average landmark points.
    
    gc.collect()
    return(img,point)



#############################################Functions for step 2####################################################


# Read all jpg images in folder.
# Reads for one sequnce video of one person
# Only used for calculating the Average Face, none of the normalised image or points are saved
def readImages(path,allPoints,numImages,pointsAvg) :
    global idx
    
    #Create array of array of images.
    imagesNorm = []
    pointsNorm = []

    #List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
       
        if filePath.endswith(".png"):
            # Read image found.
            img = cv2.imread(os.path.join(path,filePath))
            img = img[:,:320,:]

            # Convert to floating point
            img = np.float32(img)/255.0
            point = allPoints[idx]
            idx += 1
            (img,point) = coordinate_transform(point,img)
            pointsAvg = pointsAvg + point / numImages

            del img
    gc.collect()
    return(imagesNorm,pointsNorm,pointsAvg)
     
   



#############################################Functions for step 3 ####################################################

# Read points from one text file
def readPointsWarp(path) :           
    #Create an array of points.
    points = []
            
    # Read points from filePath
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((float(x), float(y)))
    file.close()
    # Store array of points           
    return points


#goes over one sequence of images
def readImagesWarp(ImagePath,TxtPath,ImagePathW,TxtPathW):
    global num_frames, frames
    # Output image
    output = np.zeros((h,w,3), np.float32())

    for filePath in sorted(os.listdir(ImagePath)):
       
        if filePath.endswith(".png"):
            fileName = os.path.splitext(filePath)[0]+"_aam.txt"
            ImagePath_img = os.path.join(ImagePath,filePath)
            TxtPath_img = os.path.join(TxtPath,fileName)
            ImagePathW_img = os.path.join(ImagePathW,filePath)
            TxtPathW_img = os.path.join(TxtPathW,fileName)

            point = readPointsWarp(TxtPath_img)
            # Read image found.
            img = cv2.imread(ImagePath_img)
            img = img[:,:320,:]

            # Convert to floating point
            img = np.float32(img)/255.0
            (imgNorm,pointNorm) = coordinate_transform(point,img)
            output += warp_one_image(imgNorm,ImagePathW_img,TxtPathW_img,pointNorm,pointsAvg,output)/num_frames
            
                
            # Add to array of images
            del img,point,imgNorm,pointNorm
            gc.collect()
    
    if not frames:
        frame_to_videos(ImagePathW,ImagePathW)
    del ImagePath,TxtPath
    gc.collect()
    return(output)




#Takes one 'normalised' image and warps it to the 'Average Face'
def warp_one_image(imgNorm,ImagePathW,TxtPathW,pointNorm,pointsAvg,output):
    # uncomment if you'd like to see centralised unwarped image
    #cv2.imwrite('face.png',imgNorm*255) 
    # Delaunay triangulation
    rect = (0, 0, int(w), int(h))
    dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))    
    # Warp input images to average image landmarks
 
    img = np.zeros((h,w,3), np.float32())
    # Transform triangles one by one
    for j in range(0, len(dt)) :
        tin = []
        tout = []
            
        for k in range(0, 3) :                
            pIn = pointNorm[dt[j][k]]
            pIn = constrainPoint(pIn, w, h)
                
            pOut = pointsAvg[dt[j][k]]
            pOut = constrainPoint(pOut, w, h)
                
            tin.append(pIn)
            tout.append(pOut)
            
            
        warpTriangle(imgNorm, img, tin, tout)

    # Add image intensities for averaging
    output = img
    img = extractor(img*255,pointNorm)
    cv2.imwrite(ImagePathW,img)
    np.savetxt(TxtPathW,pointNorm)

    del pointNorm,img
    gc.collect()
    return output



#Extracts the smallest convex hull of the normalised, warped face
def extractor(img,points):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    mask = mask.astype(np.uint8)
    points = np.array(points, np.int32)
    convexhull = cv2.convexHull(points[:66])
    cv2.fillConvexPoly(mask, convexhull, 255)
    face_extracted = cv2.bitwise_and(img,img,mask=mask)
    #inverts image, turns black background to white
    face_extracted[np.where((face_extracted==[0,0,0]).all(axis=2))] = [255,255,255]

    return face_extracted

#takes one sequence of warped images and makes a video rendering
def frame_to_videos(pathIn,pathOut):
    global fps
    name = pathIn.split('/')[-1]
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f)) and f.split('.')[-1] == 'png']
    #for sorting the file names properly
    files.sort(key = lambda x: x[5:-4])
    files.sort()
    for i in range(len(files)):
        filename=pathIn + '/'+files[i]
        #reading each files
        img = cv2.imread(filename)
        try: 
            os.remove(filename)
        except: pass
        height, width, layers = img.shape
        size = (width,height)
        
        #inserting the frames into an image array
        frame_array.append(img)
    pathOut += '/'+name+'.mp4'
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

#######################################################################################################################################              

#Initlaisng global var since couldnt do that in main
def set_global(F,f,n):
    global frames, fps, num_frames
    frames = F
    fps = f
    num_frames = 0

# for parser to convert str arg to boolean
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__' :
     #reading input arguments
     parser = argparse.ArgumentParser( prog='Average Face',prefix_chars='-')
     parser.add_argument("-Input", default='UNBC', help="put input images path")
     parser.add_argument("-Output", default='UNBC_Warped', help="put output images path")
     parser.add_argument("-Test", default=False, help='Set to true to warp images with pre computed average face', type=str2bool)
     parser.add_argument("-fps", default=29.97, help='The fps of the input frames', type=int)
     parser.add_argument("-Frames", default=True, help='Set to true if output needs to be Frames', type=str2bool)

     args = parser.parse_args()
     test = bool(args.Test)
     path = args.Input+'/'
     pathW = args.Output+'/'
     fps = args.fps
     frames = bool(args.Frames)
     set_global(frames,fps,0)

     # if not os.path.exists(pathW):
     #     os.mkdir(pathW)
     # if not os.path.exists(pathW+"Images"):
     #     os.mkdir(pathW+"Images")
     # if not os.path.exists(pathW+"AAM_landmarks"):
     #     os.mkdir(pathW+"AAM_landmarks")

   

     ##################################################Step1#########################################################################
     print("Starting Step 1 \n")
     allPoints = []
     #Just reading Points and Images, now commented
     for PeoplefilePath in sorted(os.listdir(path+'Images/')):
        if PeoplefilePath != ".DS_Store":
            for SeqfilePath in sorted(os.listdir(path+'Images/'+PeoplefilePath)):
                if SeqfilePath != ".DS_Store":
                    TxtPath = path+'AAM_landmarks/'+PeoplefilePath+'/'+SeqfilePath
                    # Read points for all images
                    if not test:
                        allPoints += readPoints(TxtPath)
                    else:
                        countFrames(TxtPath)
                    gc.collect()
     np.save('allPoints',allPoints)
     print("Step 1 done\n")



        ##################################################Step2#########################################################################
    #skip step 2 if using pre computed face
     if not test:
        print("Starting Step 2 \n")
        allPoints = np.load('allPoints.npy',mmap_mode='r').tolist()
        pointsAvg = np.array([(0,0)]* ( len(allPoints[0]) + 8 ), np.float32())

        for PeoplefilePath in sorted(os.listdir(path+'Images/')):
            if PeoplefilePath != ".DS_Store":
                for SeqfilePath in sorted(os.listdir(path+'Images/'+PeoplefilePath)):
                    if SeqfilePath != ".DS_Store":
                        ImagePath = path+'Images/'+PeoplefilePath+'/'+SeqfilePath   
                        # Read one sequence of one person
                        (imageNorm,pointNorm,pointsAvg) = readImages(ImagePath,allPoints,num_frames,pointsAvg)
                        gc.collect()

                gc.collect()
                del imageNorm,pointNorm
        np.save('AverageLandmarks',pointsAvg)
        print("Step 2 done\n")



    ##################################################Step3#########################################################################

 
     #Now actual warping

     print("Starting Step 3\n")
     pointsAvg = np.load('AverageLandmarks.npy',mmap_mode='r').tolist()
     output = np.zeros((h,w,3), np.float32())


     for PeoplefilePath in sorted(os.listdir(path+'Images/')):
        if PeoplefilePath != ".DS_Store":
            for SeqfilePath in sorted(os.listdir(path+'Images/'+PeoplefilePath)):
                if SeqfilePath != ".DS_Store":
                    ImagePath = path+'Images/'+PeoplefilePath+'/'+SeqfilePath
                    TxtPath = path+'/AAM_landmarks/'+PeoplefilePath+'/'+SeqfilePath
                    ImagePathW = pathW+'Images/'+PeoplefilePath+'/'+SeqfilePath
                    TxtPathW = pathW+'/AAM_landmarks/'+PeoplefilePath+'/'+SeqfilePath
                    # Read points for all images
                    if not os.path.exists(ImagePathW):
                        os.makedirs(ImagePathW)

                    if not os.path.exists(TxtPathW):   
                        os.makedirs(TxtPathW)

                    
                    output += readImagesWarp(ImagePath,TxtPath,ImagePathW,TxtPathW)
                    gc.collect()
     cv2.imwrite('TheAverageFace'+'.png',output*255)
     print("Step 3 done\n")

###############################################################################################################################
    

