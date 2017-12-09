# Task 1:
def HoughCircles(imageEdges, radius, votesThresh, distThresh):
    """
    implement the edge version of the circles Hough transform
    :param imageEdges: image represent edge pixels only
    :param radius: array represent the different radiuses to search circles at
    :param votesThresh: threshold represents the minimal number of votes
    required to declare a circle
    :param distThresh: threshold represents the minimal distance between the
    centers of two different circles
    :return: all the circles (center (x,y)  and radius) detected on the image
    as an Nx3 array where each row is x,y,r.
    """
    #To clean-up your detected circles and return only local-maxima circles you
    # may use the function ex3Utils.selectLocalMaxima provided in the
    # ex3utils.py file.
    #Relevant numpy functions:
    #numpy.nonzero: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.nonzero.html
    #numpy.argwhere: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.argwhere.html

    return "np"


# Task 2:
def bilateralFilter(imgNoisy, spatial_std, range_std):
    """
    WILL FILL LATER
    :param imgNoisy:
    :param spatial_std:
    :param range_std:
    :return:
    """
    return "no"
