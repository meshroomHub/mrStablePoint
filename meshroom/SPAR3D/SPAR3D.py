__version__ = "1.0"

from re import M
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class SPAR3DNodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        from pathlib import Path
        import itertools

        input_path_param = node.attribute(self._params[0])
        extension_param = node.attribute(self._params[1])

        input_path = input_path_param.value
        extension = extension_param.value
        include_suffixes = [extension.lower(), extension.upper()]

        size = 1
        if Path(input_path).is_dir():
            image_paths = list(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
            size = len(image_paths)
        elif node.attribute(self._params[0]).isLink:
            size = node.attribute(self._params[0]).getLinkParam().node.size
        
        return size


class SPAR3DBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class SPAR3D(desc.Node):
    category = "Mesh Generation"
    documentation = """This node computes a mesh from a monocular image using the StablePointAware3d deep model."""
    
    gpu = desc.Level.INTENSIVE

    size = SPAR3DNodeSize(['inputImages', 'inputExtension'])
    parallelization = SPAR3DBlockSize()

    inputs = [
        desc.File(
            name="inputImages",
            label="Input Images",
            description="Input images to estimate the depth from. Folder path or sfmData filepath",
            value="",
        ),
        desc.ChoiceParam(
            name="inputExtension",
            label="Input Extension",
            description="Extension of the input images. This will be used to determine which images are to be used if \n"
                        "a directory is provided as the input.",
            values=["jpg", "jpeg", "png", "exr"],
            value="jpg",
            exclusive=True,
        ),
        desc.IntParam(
            name="textureResolution",
            label="Texture Resolution",
            value=1024,
            description="Texture atlas resolution. Default: 1024",
            range=(256, 4096, 64),
        ),
        desc.FloatParam(
            name="foregroundRatio",
            label="Foreground Ratio",
            value=1.3,
            description="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 1.3",
            range=(0.5, 2.0, 0.01),
        ),
        desc.ChoiceParam(
            name="remeshOption",
            label="Remesh Option",
            description="Remeshing option",
            values=["none", "triangle", "quad"],
            value="none",
            exclusive=True,
        ),
        desc.ChoiceParam(
            name="reductionCountType",
            label="Reduction Count Type",
            description="Vertex count type",
            values=["keep", "vertex", "faces"],
            value="keep",
            exclusive=True,
        ),
        desc.IntParam(
            name="targetCount",
            label="Target Count",
            value=2000,
            description="Selected target count.",
            range=(100, 10000, 100),
        ),
        desc.ChoiceParam(
            name="device",
            label="Device",
            description="Model execution device",
            values=["cpu", "cuda"],
            value="cuda",
            exclusive=True,
        ),
        desc.BoolParam(
            name="lowVramMode",
            label="Low Vram Mode",
            value=False,
            description="Use low VRAM mode. SPAR3D consumes 10.5GB of VRAM by default. "
                        "This mode will reduce the VRAM consumption to roughly 7GB but in exchange "
                        "the model will be slower. Default: False",
            enabled=lambda node: node.device.value == "cuda",
        ),
        desc.IntParam(
            name="blockSize",
            label="Block Size",
            value=50,
            description="Sets the number of images to process in one chunk. If set to 0, all images are processed at once.",
            range=(0, 1000, 1),
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name='output',
            label='Output Folder',
            description="Output folder containing the computed meshes.",
            value="{nodeCacheFolder}",
        ),
    ]

    def preprocess(self, node):
        extension = node.inputExtension.value
        input_path = node.inputImages.value

        image_paths = get_image_paths_list(input_path, extension)

        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')

        self.image_paths = image_paths

    def processChunk(self, chunk):
        from spar3d.models.mesh import QUAD_REMESH_AVAILABLE, TRIANGLE_REMESH_AVAILABLE
        from spar3d.system import SPAR3D
        from spar3d.utils import foreground_crop, get_device, remove_background

        from transparent_background import Remover

        from PIL import Image

        import torch
        #from img_proc import image
        import os
        from contextlib import nullcontext
        import numpy as np
        from pathlib import Path
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.inputImages.value:
                chunk.logger.warning('No input folder given.')

            chunk_image_paths = self.image_paths[chunk.range.start:chunk.range.end]

            device = "cpu"
            if chunk.node.device.value == "cuda" and torch.cuda.is_available():
                device = "cuda"
            elif chunk.node.device.value == "cuda" and not torch.cuda.is_available():
                chunk.logger.warning('CUDA is not available, running on CPU')

            # Initialize models
            print("Loading stable point aware 3d model...")

            spar3d_model_path = os.getenv('STABLEPOINTAWARE3D_MODEL_PATH')
            model = SPAR3D.from_pretrained(
                spar3d_model_path,
                config_name="config.yaml",
                weight_name="model.safetensors",
                low_vram_mode=chunk.node.lowVramMode.value,
            )
            model.to(device)
            model.eval()

            bg_remover = Remover(device=device)
            images = []
            idx = 0

            # computation
            chunk.logger.info(f'Starting computation on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)}...')

            for idx, path in enumerate(chunk_image_paths):

                image = remove_background(Image.open(str(chunk_image_paths[idx])).convert("RGBA"), bg_remover)
                image = foreground_crop(image, chunk.node.foregroundRatio.value)
                outputDirPath = Path(chunk.node.output.value)
                image_stem = Path(chunk_image_paths[idx]).stem
                if_file_name = "input_" + str(image_stem) + ".png"
                image.save(os.path.join(outputDirPath, if_file_name))
                images = []
                images.append(image)

                vertex_count = (
                    -1
                    if chunk.node.reductionCountType.value == "keep"
                    else (
                        chunk.node.targetCount.value
                        if chunk.node.reductionCountType.value == "vertex"
                        else chunk.node.targetCount.value // 2
                    )
                )

                image = images[0:1]
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    with (
                        torch.autocast(device_type=device, dtype=torch.bfloat16)
                        if "cuda" in device
                        else nullcontext()
                    ):
                        mesh, glob_dict = model.run_image(
                            image,
                            bake_resolution=chunk.node.textureResolution.value,
                            remesh=chunk.node.remeshOption.value,
                            vertex_count=vertex_count,
                            return_points=True,
                        )
                if torch.cuda.is_available():
                    print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

                mesh_file_name = "mesh_" + str(image_stem) + ".glb"
                out_mesh_path = os.path.join(outputDirPath, mesh_file_name)
                mesh.export(out_mesh_path, include_normals=True)
                points_file_name = "points_" + str(image_stem) + ".ply"
                out_points_path = os.path.join(outputDirPath, points_file_name)
                glob_dict["point_clouds"][0].export(out_points_path)

            chunk.logger.info('SPAR3D end')
        finally:
            chunk.logManager.end()

def get_image_paths_list(input_path, extension):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO
    from pathlib import Path
    import itertools

    include_suffixes = [extension.lower(), extension.upper()]
    image_paths = []

    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
    elif Path(input_path).suffix.lower() in [".sfm", ".abc"]:
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                for id, v in views.items():
                    image_paths.append(Path(v.getImage().getImagePath()))
            image_paths.sort()
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (folder or sfmData file).")
    return image_paths
