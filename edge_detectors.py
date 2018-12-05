# Danish Waheed
# COP 5415 - FALL 2017
from PIL import Image
from pylab import *
from scipy import ndimage, misc
import cv2

# Function to get the gaussian value
def getGaussianValue(x, sigma):
    return (1/math.sqrt(2 * math.pi * sigma)) * math.pow(math.e, math.pow(-x, 2) / (2 * math.pow(sigma, 2)))

# Function to get the first derivative of the gaussian value
def getGaussianFirstDer(x, sigma):
    return (-x / math.sqrt(2 * math.pi * math.pow(sigma, 3))) * math.pow(math.e, math.pow(-x, 2) / (2 * math.pow(sigma, 2)))

# Function to get the magnitude array of IxPrime and IyPrime
def getMagnitudeArray(rows, cols, IxPrime, IyPrime):
    magArray = [[0 for y in range(cols)] for x in range(rows)]
    for a in range(0, cols):
        for b in range(0, rows):
            magArray[b][a] = math.sqrt(math.pow(IxPrime[b][a], 2) + math.pow(IyPrime[b][a], 2))
    return magArray

def nonMaximumSuppression(magArray, rows, cols, IxPrime, IyPrime):
    # We are doing -1 because we cannot compare the edge pixels beyond the image
    for y in range(1, cols - 1):
        for x in range(1, rows - 1):
            theta = math.atan2(IyPrime[x][y], IxPrime[x][y])
            currentPixel = magArray[x][y]

            # Diag 1
            if((theta >= math.pi/8 and theta <= 3*math.pi/8) or (theta >= -7*math.pi/8 and theta<= -5*math.pi/8)):
                if (currentPixel <= magArray[x - 1][y + 1] or currentPixel <= magArray[x + 1][y - 1]):
                    magArray[x][y] = 0
            # Horizontal
            elif((theta > -math.pi/8 and theta < math.pi/8) or (theta > 7*math.pi/8 and theta <= math.pi) or (theta >= -math.pi and theta < -7*math.pi/8)):
                if (currentPixel <= magArray[x][y + 1] or currentPixel <= magArray[x][y - 1]):
                    magArray[x][y] = 0
            # Diag 2
            elif((theta >= 5*math.pi/8 and theta <= 7*math.pi/8) or (theta >= -3*math.pi/8 and theta <= -math.pi/8)):
                if (currentPixel <= magArray[x - 1][y - 1] or currentPixel <= magArray[x + 1][y + 1]):
                    magArray[x][y] = 0
            # Vertical
            elif (currentPixel <= magArray[x + 1][y] or currentPixel <= magArray[x - 1][y]):
                    magArray[x][y] = 0

    return magArray


def canny_edge_detector(filename):
    # Reading in the image in color first to present on page
    I = array(Image.open(filename))
    showImageFromArray(I, 'Original image')

    # Read the image to an array as a grayscale and name it I
    I = array(Image.open(filename).convert('L'))
    print "Size of image array is "
    print I.shape
    print "Size of x axis of image is "
    print len(I[0])

    # Initializing our Gaussian masks
    G = [-2, -1, 0, 1, 2]
    Gx = [-2, -1, 0, 1, 2]
    Gy = [-2, -1, 0, 1, 2]

    # Retrieving the Sigma value from the user
    sigma = 1.5

    # Looping through all three arrays and assigning gaussian values to them
    for index in range(len(G)):
        G[index] = getGaussianValue(G[index], sigma)
        Gx[index] = getGaussianFirstDer(Gx[index], sigma)
        Gy[index] = getGaussianFirstDer(Gy[index], sigma)

    # Declaring the rows and cols to be used to find Gaussian along x and y axis of image
    rows = I.shape[0]
    cols = I.shape[1]
    print rows
    print cols

    # Declaring the Ix and Iy arrays to be colvolved with the first derivative of the Gaussian masks
    Ix = [[0 for y in range(cols)] for x in range(rows)]
    Iy = [[0 for y in range(cols)] for x in range(rows)]
    IxPrime = [[0 for y in range(cols)] for x in range(rows)]
    IyPrime = [[0 for y in range(cols)] for x in range(rows)]

    # print Ix.shape[0]
    # print Iy.shape[1]

    # The for-loop which convolves the image array I along the x-axis and stores it in image array Ix
    newArray = np.convolve(I[0], G)

    # Convolving the image  I with G along the rows to give the component image Ix
    for a in range(0, rows):
        Ix[a] = np.convolve(I[a], G)

    # Convolving along x-axis using ndimage.convolve
    # Ix = ndimage.convolve1d(I,G, 0)
    # , mode="constant", cval=0.0

    for a in range(0, cols):
        newArray = [row[a] for row in I]
        columnArray = np.convolve(newArray, G)

        for b in range(0, rows):
            Iy[b][a] = columnArray[b]

    # Convolving image array Ix with Gaussian filter Gx
    for a in range(0, rows):
        IxPrime[a] = np.convolve(Ix[a], Gx)

    # Convolving image array Iy with Gaussian filter Gy
    for a in range(0, cols):
        newArray = [row[a] for row in Iy]
        columnArray = np.convolve(newArray, Gy)

        for b in range(0, rows):
            IyPrime[b][a] = columnArray[b]

    # Computing the magnitude array of IxPrime and IyPrime
    magArray = getMagnitudeArray(rows, cols, IxPrime, IyPrime)

    # Applying non maximum suppression on the magArray
    magArray = nonMaximumSuppression(magArray, rows, cols, IxPrime, IyPrime)

    # Displaying the final image
    randomArray = np.asarray(magArray)
    img = Image.fromarray(randomArray)
    plt.imshow(img, cmap=gray())
    plt.title('Canny Edge output'), plt.xticks([]), plt.yticks([])
    plt.show()

def histogram_equalization(filename):
    # Read the original image to an array and name it I
    I = array(Image.open(filename))

    # Show the original image and its histogram
    showImageFromArray(I, 'Original image')
    showHistogramFromArray(I, 'Original histogram')

    # Call the histogram equalization function and save the
    # updated image to the array updatedImage
    updatedImage, cumulativeDistributionFunction = histogramEqualization(I)

    # Show the updated image from applying histogram equalization function
    showImageFromArray(updatedImage, 'Histogram Equalized image')

    # Show the updated image from applying the hstogram equalization function
    showHistogramFromArray(updatedImage, 'Updated histogram')


# Function to show histogram of an image array
def showHistogramFromArray(imageArray, imageTitle):
    plt.hist(imageArray.ravel(), 256, [0, 256])
    plt.title(imageTitle)
    plt.show()


# Function to show image from an image array
def showImageFromArray(imageArray, imageTitle):
    img = Image.fromarray(imageArray.astype('uint8'))
    plt.imshow(img, cmap='gray')
    plt.title(imageTitle), plt.xticks([]), plt.yticks([])
    plt.show()


# Function to perform histogram equalization using a predefined function
def histogramEqualization(imageArray):
    # Get the image histogram by calling the histogram function
    histo,bins = histogram(imageArray.flatten(),256,normed=True)
    # Call the cumulative distribution function to histogram
    cumulativeDistributionFunction = histo.cumsum() #cumulative distribution function
    # Define the number of pixels to range the function (up until 255 since greyscale)
    cumulativeDistributionFunction = 255 * cumulativeDistributionFunction / cumulativeDistributionFunction[-1] # normalization

    #Use linear interpolation to find new pixel values
    newImage = interp(imageArray.flatten(),bins[:-1],cumulativeDistributionFunction)

    return newImage.reshape(imageArray.shape), cumulativeDistributionFunction


# Function to perform gradientClipping. It calls the helper function gradientFunctionClipping.
def gradientClipping(imageArray, a, b, beta):
    # Going through the range of the x axis of the image
    for x in range(imageArray.shape[0]):
        # Going through the range of y axis of the image
        for y in range(0, imageArray.shape[1]):
            # Calling the gradient clipping function for each pixel in coordinate x,y
            imageArray[x][y] = gradientFunctionClipping(imageArray[x][y], a, b, beta)

    return imageArray


def gradientFunctionClipping(x, a, b, beta):
    # This is simply a Python implementation of the gradient function
    if 0 <= x < a:
        return 0

    if a <= x < b:
        return beta*(x - a)

    if b <= x < 256:
        return beta*(b - a)


# Range compression function uses the function provided in the slides
def rangeCompression(imageArray, c):
    for x in range(imageArray.shape[0]):

        for y in range(0, imageArray.shape[1]):
            # Python implementation of the function provided in the slides
            imageArray[x][y] = c * math.log10(1 + imageArray[x][y])

    return imageArray


# Function used to return histogram equalization
def histogram_equalizer(filename):
    # Read the original image to an array and name it I
    I = array(Image.open(filename).convert('L'))

    # Show the original image and its histogram
    showImageFromArray(I)
    showHistogramFromArray(I)

    # Call the histogram equalization function and save the
    # updated image to the array updatedImage
    updatedImage, cumulativeDistributionFunction = histogramEqualization(I)

    # Show the updated image from applying histogram equalization function
    showImageFromArray(updatedImage)

    # Show the updated image from applying the hstogram equalization function
    showHistogramFromArray(updatedImage)

# Function used to implement sobel filtering
def sobel(filename):
    # Read the original image to an array and name it I
    I = array(Image.open(filename))

    # Show the original image
    showImageFromArray(I, 'Original image')

    # Load a color image in grayscale
    img = cv2.imread(filename, 0)

    # Remove noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Generating the Sobel converted images
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # Showing the resulting image from applying Sobel filter on x-axis
    plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Showing the resulting image from applying the Sobel filter on y-axis
    plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()

# Function used to implement prewitt filtering
def prewitt(filename):
    # Read the original image to an array and name it I
    I = array(Image.open(filename).convert('L'))

    # Show the original image
    showImageFromArray(array(Image.open(filename)), 'Original image')

    prewittx = ndimage.prewitt(I, 0)
    prewitty = ndimage.prewitt(I, 1)

    plt.imshow(prewittx, cmap='gray')
    plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.imshow(prewitty, cmap='gray')
    plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])
    plt.show()

# Function to implement Laplacian gradient
def laplac(filename):
    # Read the original image to an array and name it I
    I = array(Image.open(filename))

    # Show the original image
    showImageFromArray(I, 'Original image')
    
    # Load a color image in grayscale
    img = cv2.imread(filename, 0)

    # Remove noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Convolute with proper kernels
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian image'), plt.xticks([]), plt.yticks([])

    plt.show()