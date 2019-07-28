import { ConvTranspose } from '../../../ops/conv-transpose';
import { Tensor } from '../../../tensor';
import { PoolConvUtil } from '../../../util';
import { CpuInferenceHandler } from '../inference-handler';

import { matMul2d } from './matmul';

export class CpuConvTranspose extends ConvTranspose {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const x = inputs[0];
    const w = inputs[1];
    const b = inputs.length === 3 ? inputs[2] : undefined;

    // create output tensor
    // TODO: correctly calculate output dimensions
    const outputDims = [4, 4, 4, 4];
    const y = new Tensor(outputDims, x.type);

    convTranspose2d(y, x, w, b, this.dilations, this.group, this.pads, this.strides);
    return [y];
  }
}

export function convTranspose2d(Y: Tensor, X: Tensor, W: Tensor, B: Tensor | undefined,
  dialations: ReadonlyArray<number>, group: number, pads: ReadonlyArray<number>, strides: ReadonlyArray<number>) {

}
