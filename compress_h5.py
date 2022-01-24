from PyImarisWriter import PyImarisWriter as PW
import numpy as np
from datetime import datetime
import time

class MyCallbackClass(PW.CallbackClass):

    def __init__(self):

        self.mUserDataProgress = 0

    def RecordProgress(self, progress, total_bytes_written):

        progress100 = int(progress * 100)
        if progress100 - self.mUserDataProgress >= 5:
            self.mUserDataProgress = progress100
            print('{}% Complete, Bytes written: {}'.format(self.mUserDataProgress, total_bytes_written))

class TestConfiguration:

    def __init__(self, id, title, np_type, imaris_type, compression, color_table):

        self.mId = id
        self.mTitle = title
        self.mNp_type = np_type
        self.mImaris_type = imaris_type
        self.mCompression = compression
        self.mColor_table = color_table

def get_test_configurations():
    
    configurations = []
    
    configurations.append(TestConfiguration(len(configurations), 'compression_gzip_level1', np.uint16, 'uint16', PW.eCompressionAlgorithmGzipLevel1,
                                            [PW.Color(0, 1, 1, 1), PW.Color(1, 0, 1, 1), PW.Color(1, 1, 0, 1)]))
    
    configurations.append(TestConfiguration(len(configurations), 'compression_lz4', np.uint16, 'uint16', PW.eCompressionAlgorithmLZ4,
                                            [PW.Color(0, 1, 1, 1), PW.Color(1, 0, 1, 1), PW.Color(1, 1, 0, 1)]))

    configurations.append(TestConfiguration(len(configurations), 'compression_shuffle_lz4', np.uint16, 'uint16', PW.eCompressionAlgorithmShuffleLZ4,
                                            [PW.Color(0, 1, 1, 1), PW.Color(1, 0, 1, 1), PW.Color(1, 1, 0, 1)]))

    configurations.append(TestConfiguration(len(configurations), 'compression_none', np.uint16, 'uint16', PW.eCompressionAlgorithmNone,
                                            [PW.Color(0, 1, 1, 1), PW.Color(1, 0, 1, 1), PW.Color(1, 1, 0, 1)]))

    return configurations

def run(configuration, np_data, cores):

    image_size = PW.ImageSize(x = np_data.shape[0], y = np_data.shape[1], z = np_data.shape[2], c = 1, t = 1)
    dimension_sequence = PW.DimensionSequence('x', 'y', 'z', 'c', 't')
    block_size = image_size
    sample_size = PW.ImageSize(x = 1, y = 1, z = 1, c = 1, t = 1)
    num_voxels = image_size.x*image_size.y*image_size.z*image_size.c*image_size.t
    num_voxels_per_block = block_size.x*block_size.y*block_size.z*block_size.c*block_size.t
    output_filename = f'data_{configuration.mTitle}.h5'

    options = PW.Options()
    options.mNumberOfThreads = cores
    options.mCompressionAlgorithmType = configuration.mCompression
    options.mEnableLogProgress = True

    application_name = 'PyImarisWriter'
    application_version = '1.0.0'

    callback_class = MyCallbackClass()
    converter = PW.ImageConverter(configuration.mImaris_type, image_size, sample_size, dimension_sequence, block_size,
                                  output_filename, options, application_name, application_version, callback_class)
    
    num_blocks = image_size/block_size

    start_time = time.time()

    block_index = PW.ImageSize()
    for c in range(num_blocks.c):
        block_index.c = c
        for t in range(num_blocks.t):
            block_index.t = t
            for z in range(num_blocks.z):
                block_index.z = z
                for y in range(num_blocks.y):
                    block_index.y = y
                    for x in range(num_blocks.x):
                        block_index.x = x
                        if converter.NeedCopyBlock(block_index):
                            converter.CopyBlock(np_data, block_index)

    adjust_color_range = True
    image_extents = PW.ImageExtents(0, 0, 0, image_size.x, image_size.y, image_size.z)
    parameters = PW.Parameters()
    parameters.set_channel_name(0, configuration.mTitle)
    time_infos = [datetime.today()]
    color_infos = [PW.ColorInfo() for _ in range(image_size.c)]
    color_infos[0].set_color_table(configuration.mColor_table) 
    converter.Finish(image_extents, parameters, time_infos, color_infos, adjust_color_range)
    converter.Destroy()
    
    print('{} MB/sec/core'.format(image_size.x*image_size.y*image_size.z*image_size.c*image_size.t*2/1e6/(time.time()-start_time)/options.mNumberOfThreads))
    print('Wrote {} to {}'.format(configuration.mTitle, output_filename))

def main():

    camX = 2048
    camY = 2048
    nFrames = 1000
    nCores = 4

    np_data = np.random.poisson(size = (camX, camY, nFrames))
    np_data = np_data/np.max(np_data[:])*1000

    configurations = get_test_configurations()

    for test_config in configurations:
        run(test_config, np_data.astype(test_config.mNp_type), nCores)

if __name__ == "__main__":

    main()