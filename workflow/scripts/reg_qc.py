import base64
import os
import re
from glob import glob
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.datasets import load_mni152_template
from svgutils.compose import Unit
from svgutils.transform import GroupElement, SVGFigure, fromstring

	
def svg2str(display_object, dpi):
    """Serialize a nilearn display object to string."""

    image_buf = StringIO()
    display_object.frame_axes.figure.savefig(
        image_buf, dpi=dpi, format="svg", facecolor="k", edgecolor="k"
    )
    return image_buf.getvalue()


def extract_svg(display_object, dpi=250):
    """Remove the preamble of the svg files generated with nilearn."""
    image_svg = svg2str(display_object, dpi)

    image_svg = re.sub(' height="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(' width="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(
        " viewBox", ' preseveAspectRation="xMidYMid meet" viewBox', image_svg, count=1
    )
    start_tag = "<svg "
    start_idx = image_svg.find(start_tag)
    end_tag = "</svg>"
    end_idx = image_svg.rfind(end_tag)

    # rfind gives the start index of the substr. We want this substr
    # included in our return value so we add its length to the index.
    end_idx += len(end_tag)

    return image_svg[start_idx:end_idx]


def clean_svg(fg_svgs, bg_svgs, ref=0):
    # Find and replace the figure_1 id.
    svgs = bg_svgs + fg_svgs
    roots = [f.getroot() for f in svgs]

    sizes = []
    for f in svgs:
        viewbox = [float(v) for v in f.root.get("viewBox").split(" ")]
        width = int(viewbox[2])
        height = int(viewbox[3])
        sizes.append((width, height))
    nsvgs = len([bg_svgs])

    sizes = np.array(sizes)

    # Calculate the scale to fit all widths
    width = sizes[ref, 0]
    scales = width / sizes[:, 0]
    heights = sizes[:, 1] * scales

    # Compose the views panel: total size is the width of
    # any element (used the first here) and the sum of heights
    fig = SVGFigure(Unit(f"{width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

    yoffset = 0
    for i, r in enumerate(roots):
        r.moveto(0, yoffset, scale_x=scales[i])
        if i == (nsvgs - 1):
            yoffset = 0
        else:
            yoffset += heights[i]

    # Group background and foreground panels in two groups
    if fg_svgs:
        newroots = [
            GroupElement(roots[:nsvgs], {"class": "background-svg"}),
            GroupElement(roots[nsvgs:], {"class": "foreground-svg"}),
        ]
    else:
        newroots = roots

    fig.append(newroots)
    fig.root.attrib.pop("width", None)
    fig.root.attrib.pop("height", None)
    fig.root.set("preserveAspectRatio", "xMidYMid meet")

    with TemporaryDirectory() as tmpdirname:
        out_file = Path(tmpdirname) / "tmp.svg"
        fig.save(str(out_file))
        # Post processing
        svg = out_file.read_text().splitlines()

    # Remove <?xml... line
    if svg[0].startswith("<?xml"):
        svg = svg[1:]

    # Add styles for the flicker animation
    if fg_svgs:
        svg.insert(
            2,
            """\
<style type="text/css">
@keyframes flickerAnimation%s { 0%% {opacity: 1;} 100%% { opacity:0; }}
.foreground-svg { animation: 1s ease-in-out 0s alternate none infinite running flickerAnimation%s;}
.foreground-svg:hover { animation-play-state: running;}
</style>"""
            % tuple([uuid4()] * 2),
        )

    return svg


def sorted_nicely(data, reverse=False):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key, reverse=reverse)
   

def load_and_format_nifti(path):
    img = nib.load(path)
    return nib.Nifti1Image(
        img.get_fdata().astype(np.float32),
        header=img.header,
        affine=img.affine
    )

def display_img_for_modes(img, modes, cut_coords, plot_args):
    svgs = []
    for mode in modes:
        display = plotting.plot_anat(
            img,
            display_mode=mode,
            draw_cross=False,
            cut_coords=cut_coords,
            **plot_args
        )
        svgs.append(fromstring(extract_svg(display, 300)))
        display.close()
    return svgs

def generate_html_for_visualizations(img1, img2, plot_args, cut_coords):
    fg_svgs = display_img_for_modes(img1, ["x", "y", "z"], cut_coords, plot_args)
    bg_svgs = display_img_for_modes(img2, ["x", "y", "z"], cut_coords, plot_args)
    return "\n".join(clean_svg(fg, bg) for fg, bg in zip(fg_svgs, bg_svgs))

def output_html(preSLAH, postSLAH, ref, preMNI, output_html):
    nongad_rigid = nib.load(preSLAH)
    nongad_rigid = nib.Nifti1Image(
        nongad_rigid.get_fdata().astype(np.float32),
        header=nongad_rigid.header,
        affine=nongad_rigid.affine,
    )
    plot_args_ref = {"dim": -0.5}

    display_x = plotting.plot_anat(
        nongad_rigid,
        display_mode="x",
        draw_cross=False,
        cut_coords=(-60, -40, 0, 20, 40, 60),
        **plot_args_ref,
    )
    fg_x_svgs = [fromstring(extract_svg(display_x, 300))]
    display_x.close()

    display_y = plotting.plot_anat(
        nongad_rigid,
        display_mode="y",
        draw_cross=False,
        cut_coords=(-40, -20, 0, 20, 40, 60),
        **plot_args_ref,
    )
    fg_y_svgs = [fromstring(extract_svg(display_y, 300))]
    display_y.close()

    display_z = plotting.plot_anat(
        nongad_rigid,
        display_mode="z",
        draw_cross=False,
        cut_coords=(-40, -20, 0, 20, 40, 60),
        **plot_args_ref,
    )
    fg_z_svgs = [fromstring(extract_svg(display_z, 300))]
    display_z.close()

    gad_img = nib.load(postSLAH)
    gad_img = nib.Nifti1Image(
        gad_img.get_fdata().astype(np.float32),
        header=gad_img.header,
        affine=gad_img.affine,
    )

    display_x = plotting.plot_anat(
        gad_img,
        display_mode="x",
        draw_cross=False,
        cut_coords=(-60, -40, 0, 20, 40, 60),
        **plot_args_ref,
    )
    bg_x_svgs = [fromstring(extract_svg(display_x, 300))]
    display_x.close()

    display_y = plotting.plot_anat(
        gad_img,
        display_mode="y",
        draw_cross=False,
        cut_coords=(-40, -20, 0, 20, 40, 60),
        **plot_args_ref,
    )
    bg_y_svgs = [fromstring(extract_svg(display_y, 300))]
    display_y.close()

    display_z = plotting.plot_anat(
        gad_img,
        display_mode="z",
        draw_cross=False,
        cut_coords=(-40, -20, 0, 20, 40, 60),
        **plot_args_ref,
    )
    bg_z_svgs = [fromstring(extract_svg(display_z, 300))]
    display_z.close()

    final_svg_rigid_x = "\n".join(clean_svg(fg_x_svgs, bg_x_svgs))
    final_svg_rigid_y = "\n".join(clean_svg(fg_y_svgs, bg_y_svgs))
    final_svg_rigid_z = "\n".join(clean_svg(fg_z_svgs, bg_z_svgs))

    anat_params = {
        "vmin": nongad_rigid.get_fdata(dtype="float32").min(),
        "vmax": nongad_rigid.get_fdata(dtype="float32").max(),
        "cmap": plt.cm.gray,
        "interpolation": "none",
        "draw_cross": False,
    }

    display = plotting.plot_anat(nongad_rigid, **anat_params)
    display.add_contours(gad_img, colors="r", alpha=0.7, linewidths=0.8)

    tmpfile = BytesIO()
    display.savefig(tmpfile, dpi=300)
    display.close()
    tmpfile.seek(0)
    encoded_rigid = base64.b64encode(tmpfile.getvalue())

    nongad_affine = nib.load(ref)
    nongad_affine = nib.Nifti1Image(
        nongad_affine.get_fdata().astype(np.float32),
        header=nongad_affine.header,
        affine=nongad_affine.affine,
    )

    display_x = plotting.plot_anat(
        nongad_affine,
        display_mode="x",
        draw_cross=False,
        cut_coords=(-60, -40, 0, 20, 40, 60),
        **plot_args_ref,
    )
    fg_x_svgs = [fromstring(extract_svg(display_x, 300))]
    display_x.close()

    display_y = plotting.plot_anat(
        nongad_affine,
        display_mode="y",
        draw_cross=False,
        cut_coords=(-40, -20, 0, 20, 40, 60),
        **plot_args_ref,
    )
    fg_y_svgs = [fromstring(extract_svg(display_y, 300))]
    display_y.close()

    display_z = plotting.plot_anat(
        nongad_affine,
        display_mode="z",
        draw_cross=False,
        cut_coords=(-40, -20, 0, 20, 40, 60),
        **plot_args_ref,
    )
    fg_z_svgs = [fromstring(extract_svg(display_z, 300))]
    display_z.close()

    gad_img = nib.load(preMNI)
    gad_img = nib.Nifti1Image(
        gad_img.get_fdata().astype(np.float32),
        header=gad_img.header,
        affine=gad_img.affine,
    )

    display_x = plotting.plot_anat(
        gad_img,
        display_mode="x",
        draw_cross=False,
        cut_coords=(-60, -40, 0, 20, 40, 60),
        **plot_args_ref,
    )
    bg_x_svgs = [fromstring(extract_svg(display_x, 300))]
    display_x.close()

    display_y = plotting.plot_anat(
        gad_img,
        display_mode="y",
        draw_cross=False,
        cut_coords=(-40, -20, 0, 20, 40, 60),
        **plot_args_ref,
    )
    bg_y_svgs = [fromstring(extract_svg(display_y, 300))]
    display_y.close()

    display_z = plotting.plot_anat(
        gad_img,
        display_mode="z",
        draw_cross=False,
        cut_coords=(-40, -20, 0, 20, 40, 60),
        **plot_args_ref,
    )
    bg_z_svgs = [fromstring(extract_svg(display_z, 300))]
    display_z.close()

    final_svg_affine_x = "\n".join(clean_svg(fg_x_svgs, bg_x_svgs))
    final_svg_affine_y = "\n".join(clean_svg(fg_y_svgs, bg_y_svgs))
    final_svg_affine_z = "\n".join(clean_svg(fg_z_svgs, bg_z_svgs))

    anat_params = {
        "vmin": nongad_affine.get_fdata(dtype="float32").min(),
        "vmax": nongad_affine.get_fdata(dtype="float32").max(),
        "cmap": plt.cm.gray,
        "interpolation": "none",
        "draw_cross": False,
    }

    display = plotting.plot_anat(nongad_affine, **anat_params)
    display.add_contours(gad_img, colors="r", alpha=0.7, linewidths=0.8)

    tmpfile_affine = BytesIO()
    display.savefig(tmpfile_affine, dpi=300)
    display.close()
    tmpfile_affine.seek(0)
    encoded_affine = base64.b64encode(tmpfile_affine.getvalue())

    # Writing to an HTML file
    with open(output_html, 'w') as f:
        f.write('<html>')
        f.write('<body>')

        f.write('<h2>Rigid Registration:</h2>')
        f.write('<img src="data:image/png;base64,{}">'.format(encoded_rigid.decode('utf-8')))
        f.write('<h3>Axis X:</h3>')
        f.write(final_svg_rigid_x)
        f.write('<h3>Axis Y:</h3>')
        f.write(final_svg_rigid_y)
        f.write('<h3>Axis Z:</h3>')
        f.write(final_svg_rigid_z)

        f.write('<h2>Affine Registration:</h2>')
        f.write('<img src="data:image/png;base64,{}">'.format(encoded_affine.decode('utf-8')))
        f.write('<h3>Axis X:</h3>')
        f.write(final_svg_affine_x)
        f.write('<h3>Axis Y:</h3>')
        f.write(final_svg_affine_y)
        f.write('<h3>Axis Z:</h3>')
        f.write(final_svg_affine_z)

        f.write('</body>')
        f.write('</html>')

    return output_html

   


output_html(snakemake.input.image_pre ,snakemake.input.image_post , snakemake.input.MNI_temp, snakemake.input.mni_image, snakemake.output.html)


