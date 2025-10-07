from src.feature_extraction import SURFExtractor
import cv2

def run(img_path: str = "data/images/abandoned_ship_s_000004.png"):
    surf = SURFExtractor(debug=True)

    img = cv2.imread(img_path)
    surf.extract_features(img, "abandoned_ship_s_000004.png")


if __name__ == '__main__':
    run()