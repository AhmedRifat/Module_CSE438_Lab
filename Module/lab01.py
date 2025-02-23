# Define the Question class
class Question:
    def __init__(self, question_number, solution_code):
        self.question_number = question_number
        self.solution_code = solution_code

    def solution(self):
        print(f"Solution for Question {self.question_number}:\n")
        print(self.solution_code)

# Define solution codes for each question
q1_solution_code = """
def load_and_display_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image from BGR to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image using matplotlib
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide the axes
    plt.show()
"""

q2_solution_code = """
def apply_gaussian_blur(image_path, kernel_size=(5, 5)):
    # Load the image
    image = cv2.imread(image_path)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    
    # Convert to RGB for display
    blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
    
    # Display the blurred image
    plt.imshow(blurred_image_rgb)
    plt.axis('off')
    plt.show()
"""

q3_solution_code = """
def apply_canny_edge_detection(image_path, low_threshold=50, high_threshold=150):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold)
    
    # Display the edges
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.show()
"""

q4_solution_code = """
def apply_sobel_edge_detection(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Display the Sobel edges
    plt.imshow(sobel_combined, cmap='gray')
    plt.axis('off')
    plt.show()
"""

q5_solution_code = """
def read_video_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Read and display frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
"""

q6_solution_code = """
def resize_image(image_path, scale_factor=0.5):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the image
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    
    # Convert to RGB for display
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Display the resized image
    plt.imshow(resized_image_rgb)
    plt.axis('off')
    plt.show()
"""

q7_solution_code = """
def draw_shapes_and_text(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Draw a rectangle
    cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 2)
    
    # Draw a circle
    cv2.circle(image, (300, 300), 100, (255, 0, 0), 2)
    
    # Add text
    cv2.putText(image, 'OpenCV', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
"""

q8_solution_code = """
def apply_essential_methods(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    # Display the binary image
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    plt.show()
"""

q9_solution_code = """
def apply_image_transformations(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Rotate the image by 45 degrees
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    
    # Convert to RGB for display
    rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
    
    # Display the rotated image
    plt.imshow(rotated_image_rgb)
    plt.axis('off')
    plt.show()
"""

q10_solution_code = """
def detect_contours(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
    
    # Display the image with contours
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
"""

# Create instances for each question
q_1 = Question(question_number=1, solution_code=q1_solution_code)
q_2 = Question(question_number=2, solution_code=q2_solution_code)
q_3 = Question(question_number=3, solution_code=q3_solution_code)
q_4 = Question(question_number=4, solution_code=q4_solution_code)
q_5 = Question(question_number=5, solution_code=q5_solution_code)
q_6 = Question(question_number=6, solution_code=q6_solution_code)
q_7 = Question(question_number=7, solution_code=q7_solution_code)
q_8 = Question(question_number=8, solution_code=q8_solution_code)
q_9 = Question(question_number=9, solution_code=q9_solution_code)
q_10 = Question(question_number=10, solution_code=q10_solution_code)
