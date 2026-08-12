# tttrkit

**Tools for reading, streaming, and reconstructing images from PicoQuant TTTR data.**

`tttrkit` is a Python toolkit for working with time-tagged time-resolved (TTTR) data acquired with PicoQuant instrumentation. It provides tools to read TTTR data, process it in chunks, and reconstruct images (and more) according to user-defined scanning schemes.

The processing is designed around **chunk-wise streaming**, allowing large TTTR datasets to be processed without loading the complete acquisition into memory.

## Features

* Read and process PicoQuant TTTR data
* Chunk-wise streaming of TTTR records
* Chunk-wise image reconstruction
* Configurable scanning schemes and acquisition configurations
* Support for non-standard and custom scanning patterns
* Line accumulation and repeated/sequence-based acquisitions
* Utilities for FLIM data processing
* Phasor analysis utilities
* Python API suitable for interactive analysis and custom processing pipelines

## Processing model

`tttrkit` processes TTTR data **chunk-wise**:

The complete acquisition does not need to be loaded into memory at once. This is particularly useful for large time-resolved imaging datasets where the images, histograms, phasors, etc. are reconstructed by iteration over chunks of defined size.

The chunk-wise architecture also allows processing to be combined with custom analysis or downstream pipelines.

## Scanning schemes

A central concept in `tttrkit` is the **scanning scheme** (or acquisition configuration).

This allows `tttrkit` to handle acquisition strategies such as:

* standard raster scanning
* line accumulation
* line sequences (i.e, different acuisition settings altered between lines)
* ...

The intention is to keep the reconstruction logic independent of a particular microscope's scanning implementation.

## FLIM

`tttrkit` includes utilities for working with fluorescence lifetime imaging microscopy (FLIM) data.

Depending on the acquisition and reconstruction configuration, the resulting data can be used for lifetime analysis and visualization.

## Phasor analysis

The package also contains utilities for **phasor analysis of FLIM data**, providing a frequency-domain representation of fluorescence lifetime information.

These utilities are intended to facilitate analysis and visualization of FLIM datasets reconstructed from TTTR acquisitions.

## Installation

Install the package with:

```bash
pip install tttrkit
```

## Basic usage

See the example notebooks for basic usage.

## Development

Development takes place on GitHub.

For issues, feature requests, and contributions, please use the project's GitHub repository.

## Status

`tttrkit` is under active development. APIs may change between versions.

## License

[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)