import asyncio
import typing

import ezmsg.core as ez
from ezmsg.util.generator import GenState
from ezmsg.util.messages.axisarray import AxisArray

from .iterator import NWBAxisArrayIterator, NWBIteratorSettings


class NWBIteratorUnit(ez.Unit):
    STATE = GenState
    SETTINGS = NWBIteratorSettings

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
    OUTPUT_TERM = ez.OutputStream(typing.Any)

    def initialize(self) -> None:
        self.construct_generator()

    def construct_generator(self):
        self.STATE.gen = NWBAxisArrayIterator(
            settings=self.SETTINGS,
        )

    @ez.publisher(OUTPUT_SIGNAL)
    async def pub_chunk(self) -> typing.AsyncGenerator:
        for msg in self.STATE.gen:
            yield self.OUTPUT_SIGNAL, msg
            await asyncio.sleep(0)

        ez.logger.debug(f"File ({self.SETTINGS.filepath}) exhausted.")
        if self.SETTINGS.self_terminating:
            raise ez.NormalTermination
        yield self.OUTPUT_TERM, ez.Flag
