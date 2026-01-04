# Haptic-PBR-Maps
### Haptic PBR Map generator: Giving the PBR maps a new dimensions of haptic information


## Proof-of-Concept Software Model
The software accepts multimodal optional input and generate hPBR maps. ( In latest version it's text and image only for demonstration.)
By connecting exisiting Computer Vision AI models for different tasks such as Material Segmenation and PBR maps generations,
with material haptic properties database.

![Software model](https://github.com/gilkzxc/Haptic-PBR-Maps/blob/main/images/software%20model.png)

## hPBR File Structure
The new protocol extends the PBR tile maps standard by adding material properties maps that represents Haptic information.
All maps are stored in binary format for easy extraction in lossless form into the desired data structures of the user.

![hPBR File Structure](https://github.com/gilkzxc/Haptic-PBR-Maps/blob/main/images/hpbr%20map%20structure.png)

The file begins with a header that indicates that this is a Haptic PBR file and it’s version of the protocol, followed by chunks of maps as inspired by the structure of PNG. The file ends with a sha256 hash, generated from file data, to ensure that the file isn’t corrupted.
Each chunk is a PBR tile map or a property map. The chunk consists of the following parts in the following order:
*	Tag (4 Bytes) – A 4 charceter long identifier wether the map is Numpy ndarray or a PIL Image.Image object.
*	Chunk length (4 Bytes) – The size in bytes of the payload and of the checksum.
*	Name length and Name – The name of the map ( e.g. Possion Ratio, Young’s Modulus or Height) and the length of the name, so each map can have an insightful name attached in mostly it’s readable     compistion.
*	Payload – The binary data of the map.
*	Chunk checksum – CRC32 of the payload, to ensure the map itself isn’t corrupted.


## Reference
* **Vecchio, Giuseppe.** “*StableMaterials: Enhancing Diversity in Material Generation via Semi-Supervised Learning*.” *arXiv preprint* [arXiv:2406.09293](https://arxiv.org/abs/2406.09293), 2024.
* **Upchurch, Paul and Niu, Ransen.** “*A Dense Material Segmentation Dataset for Indoor and Outdoor Scene Parsing*.” In *European Conference on Computer Vision (ECCV)* [arXiv:2207.10614](https://arxiv.org/abs/2207.10614), 2022.
* **Vecchio, Giuseppe and Deschaintre, Valentin.** “*MatSynth: A Modern PBR Materials Datase*.” In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* [arXiv:2401.06056](https://arxiv.org/abs/2401.06056), 2024.

