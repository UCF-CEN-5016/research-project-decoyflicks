from IPython.display import Image, display

def show_image_from_url(url: str) -> Image:
    """Create and display an IPython Image from a URL."""
    img = Image(url)
    display(img)
    return img

def main():
    IMAGE_URL = "https://user-images.githubusercontent.com/114388973/230285747-90e0b9d1-6b8b-49f7-962c-2233cc0d2489.png"
    show_image_from_url(IMAGE_URL)

if __name__ == "__main__":
    main()