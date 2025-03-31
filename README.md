# PlumeDEBuG: Plume-Data Empowered Bubble Generator

Welcome to **PlumeDEBuG**! This Python project generates synthetic bubble images for various scientific and engineering applications, such as fluid dynamics and image processing. Using random bubble placement algorithms and a custom quadtree-based collision detection system, **PlumeDEBuG** creates realistic images of bubbles in a defined region of interest (ROI), empowered by plume-data.

## Features

- **Image Generation**: Create synthetic bubble images with adjustable parameters (bubble count, void fraction, etc.).
- **Quadtree Collision Detection**: Efficient placement of bubbles with collision checks using a quadtree.
- **Distribution Sampling**: Supports Gaussian, Weibull, Lognormal, and constant distribution types for bubble sizes.
- **Filters and Effects**: Apply filters like feathering, bilateral, and Gaussian blur to enhance image quality.
- **Data Caching**: Cache bubble data from SQLite to speed up subsequent runs.
- **Analysis**: Includes tools to plot histograms and perform K-S tests on bubble diameters.

## Requirements

To run this project, you’ll need the following libraries:

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- SciPy
- Matplotlib
- TQDM
- SQLite
- ConfigParser

You can install the necessary dependencies using `pip`:

```bash
pip install opencv-python numpy scipy matplotlib tqdm
# PlumeDEBuG: Plume-Data Empowered Bubble Generator
```
Welcome to **PlumeDEBuG**! This Python project generates synthetic bubble images for various scientific and engineering applications, such as fluid dynamics and image processing. Using random bubble placement algorithms and a custom quadtree-based collision detection system, **PlumeDEBuG** creates realistic images of bubbles in a defined region of interest (ROI), empowered by plume-data.

## Features

- **Image Generation**: Create synthetic bubble images with adjustable parameters (bubble count, void fraction, etc.).
- **Quadtree Collision Detection**: Efficient placement of bubbles with collision checks using a quadtree.
- **Distribution Sampling**: Supports Gaussian, Weibull, Lognormal, and constant distribution types for bubble sizes.
- **Filters and Effects**: Apply filters like feathering, bilateral, and Gaussian blur to enhance image quality.
- **Data Caching**: Cache bubble data from SQLite to speed up subsequent runs.
- **Analysis**: Includes tools to plot histograms and perform K-S tests on bubble diameters.

## Requirements

To run this project, you’ll need the following libraries:

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- SciPy
- Matplotlib
- TQDM
- SQLite
- ConfigParser

You can install the necessary dependencies using `pip`:

```bash
pip install opencv-python numpy scipy matplotlib tqdm
```

## How to Use
Configure config.ini: Adjust the settings in config.ini to suit your needs. The key parameters include:

num_synthetic_images: The number of synthetic images to generate.

target_void_fraction: The desired void fraction for bubble placement.

max_bubbles_per_image: Maximum number of bubbles per image.

output_dir: The directory where images will be saved.

Run the Script: Execute the Python script:

```bash

python plume_debug.py
```
This will generate synthetic bubble images and store them in the specified output directory.

Visualize Results: You can view the generated images and analyze the bubble size distribution. The script will also perform a K-S test to check the fit to the desired distribution.

## Example Output
Images: The generated images will be saved in the format synth_XXXX.png.

Annotations: JSON files containing bubble annotations (bounding boxes, masks, areas, etc.) will be saved alongside the images.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Feel free to reach out if you have any questions or need support:

Email: [xuchen.ying@missouri.edu]

GitHub: [[Github Link](https://github.com/Schuetzen/PlumeDEBuG)]

Happy coding!