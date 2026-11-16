import asyncio
import math
import typing

import ezmsg.core as ez
from ezmsg.baseproc.units import BaseProducerUnit
from ezmsg.util.messages.axisarray import AxisArray

from .iterator import NWBAxisArrayIterator, NWBIteratorSettings


class NWBIteratorUnit(BaseProducerUnit[NWBIteratorSettings, AxisArray, NWBAxisArrayIterator]):
    SETTINGS = NWBIteratorSettings

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
    OUTPUT_TERM = ez.OutputStream(typing.Any)

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        segments = {}
        while True:
            out = await self.producer.__acall__()
            if out is not None:
                # An empty chunk carries nothing, but an empty end_of_stream marker starts a new segment.
                if math.prod(out.data.shape) > 0 or segments.get(out.key, out.segment) != out.segment:
                    segments[out.key] = out.segment
                    yield self.OUTPUT_SIGNAL, out
                await asyncio.sleep(0)
            elif self.producer.exhausted:
                break

        ez.logger.debug(f"File ({self.SETTINGS.filepath}) exhausted.")
        if self.SETTINGS.self_terminating:
            raise ez.NormalTermination
        yield self.OUTPUT_TERM, ez.Flag
