import math

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.fdat_rd import FDATRD, SampleMods3


class FDATRDArch(Architecture[FDATRD]):
    def __init__(self):
        super().__init__(
            id="FDATRD",
            detect=KeyCondition.has_any(
                KeyCondition.has_all(
                    "conv_first.weight",
                    "groups.0.blocks.0.attn.bias",
                    "groups.0.blocks.2.attn.dictionary",
                    "groups.0.blocks.0.inter.cg.1.weight",
                    "groups.0.blocks.0.ffn.fc1.weight",
                    "groups.0.blocks.0.n1.weight",
                    "upsampler.MetaUpsample",
                ),
                KeyCondition.has_all(
                    "conv_first.1.weight",
                    "groups.0.blocks.0.attn.bias",
                    "groups.0.blocks.2.attn.dictionary",
                    "groups.0.blocks.0.inter.cg.1.weight",
                    "groups.0.blocks.0.ffn.fc1.weight",
                    "groups.0.blocks.0.n1.weight",
                    "upsampler.MetaUpsample",
                ),
            ),
        )

    def load(self, state_dict: StateDict) -> ImageModelDescriptor[FDATRD]:
        _, upsampler_index, scale, embed_dim, num_out_ch, mid_dim, _ = state_dict[
            "upsampler.MetaUpsample"
        ].tolist()
        upsampler_type = list(SampleMods3.__args__)[upsampler_index]

        if "conv_first.1.weight" in state_dict:
            num_in_ch = num_out_ch
            scale = 4 // math.isqrt(
                state_dict["conv_first.1.weight"].shape[1] // num_in_ch
            )
            unshuffle_mod = True
        else:
            unshuffle_mod = False
            num_in_ch = state_dict["conv_first.weight"].shape[1]

        num_groups = get_seq_len(state_dict, "groups")
        num_blocks = get_seq_len(state_dict, "groups.0.blocks")

        def btype(i: int) -> str:
            b = f"groups.0.blocks.{i}.attn."
            if b + "dictionary" in state_dict:
                return "dictionary"
            if b + "temp" in state_dict:
                return "channel"
            return "spatial"

        types = [btype(i) for i in range(num_blocks)]
        period = next(
            p
            for p in range(1, num_blocks + 1)
            if num_blocks % p == 0 and all(types[i] == types[i % p] for i in range(num_blocks))
        )
        group_block_pattern = types[:period]
        depth_per_group = num_blocks // period

        num_heads = state_dict["groups.0.blocks.0.attn.bias"].shape[0]
        # split_size factorization is not stored; (10, 30) is the standard split.
        split_size = (10, 30)
        dict_idx = types.index("dictionary")
        num_dict_tokens = state_dict[
            f"groups.0.blocks.{dict_idx}.attn.dictionary"
        ].shape[0]
        ffn_expansion_ratio = float(
            state_dict["groups.0.blocks.0.ffn.fc1.weight"].shape[0] / embed_dim
        )
        aim_reduction_ratio = (
            embed_dim // state_dict["groups.0.blocks.0.inter.cg.1.weight"].shape[0]
        )

        model = FDATRD(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            embed_dim=embed_dim,
            num_groups=num_groups,
            depth_per_group=depth_per_group,
            num_heads=num_heads,
            split_size=split_size,
            num_dict_tokens=num_dict_tokens,
            ffn_expansion_ratio=ffn_expansion_ratio,
            aim_reduction_ratio=aim_reduction_ratio,
            group_block_pattern=group_block_pattern,
            upsampler_type=upsampler_type,
            mid_dim=mid_dim,
            img_range=1.0,
            unshuffle_mod=unshuffle_mod,
        )

        sizes = {96: "small", 128: "medium", 192: "large"}
        tags = [f"{embed_dim}dim", upsampler_type]
        if embed_dim in sizes:
            tags.append(sizes[embed_dim])
        if unshuffle_mod:
            tags.append("unshuffle")

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=num_in_ch,
            output_channels=num_out_ch,
        )


__all__ = ["FDATRDArch", "FDATRD"]
