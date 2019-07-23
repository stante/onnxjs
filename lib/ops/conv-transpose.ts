import { Attribute } from '../attribute';
import { InferenceHandler } from '../backend';
import { Operator } from '../operators';
import { Tensor } from '../tensor';

export abstract class ConvTranspose implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[] | Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.autoPad = attributes.getString('auto_pad', 'NOT_SET');
    this.dilations = attributes.getInts('dilations', [1, 1]);
    this.group = attributes.getInt('group', 1);
    this.kernelShape = attributes.getInts('kernel_shape', []);
    this.outputPadding = attributes.getInts('output_padding', [0, 0, 0, 0]);
    this.outputShape = attributes.getInts('output_shape', []);
    this.pads = attributes.getInts('pads', [0, 0, 0, 0]);
    this.strides = attributes.getInts('strides', [1, 1]);
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || (inputs.length !== 2 && inputs.length !== 3)) {
      return false;
    }

    // Only support for 2D ConvTranspose
    if (inputs[0].dims.length !== 4 || inputs[1].dims.length !== 4) {
      return false;
    }

    const dataChannel = inputs[0].dims[1];
    const filterInChannel = inputs[1].dims[1] * this.group;
    if (dataChannel !== filterInChannel) {
      return false;
    }

    if (inputs.length === 3 && (inputs[2].dims.length !== 1) && (inputs[1].dims[0] !== inputs[2].dims[0])) {
      return false;
    }

    const spatialRank = inputs[0].dims.length - 2;
    // Wrong dialations dimension
    if (this.dilations.length !== spatialRank) {
      return false;
    }

    // Wrong kernel dimension
    if (this.kernelShape.length !== spatialRank) {
      return false;
    }

    // Wrong padding dimension
    if (this.pads.length !== spatialRank) {
      return false;
    }

    if (this.kernelShape.length !== 0 && this.kernelShape.length === inputs[1].dims.length - 2) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type != 'float32' || inputs[1].type != 'float32') {
      return false;
    }

    if (inputs.length === 3 && inputs[2].type != 'float32') {
      return false;
    }

    return true;
  }

  protected autoPad: string;
  protected dilations: number[];
  protected group: number;
  protected kernelShape: number[];
  protected outputPadding: number[];
  protected outputShape: number[];
  protected pads: number[];
  protected strides: number[];
}
