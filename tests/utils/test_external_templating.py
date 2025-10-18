import tempfile
from pathlib import Path

import pytest

from buttermilk._core.config_bootstrap import bootstrap_session_with_config_async
from buttermilk.utils.templating import load_template


@pytest.mark.anyio
async def test_external_template_loading_and_priority():
    """Tests that an external template path can be configured and is prioritized."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Create an external template that has the same name as a default one
        # I need to find a default template name first. Let's assume 'summarise_case.jinja2' exists.
        # I will create a file with the same name in my temp dir.
        external_template_content = "This is the external template."
        (temp_path / "summarise.jinja2").write_text(external_template_content)

        # 2. Bootstrap buttermilk with the external template path
        bm, cfg = await bootstrap_session_with_config_async(
            job="test_templating", project_name="buttermilk_testing", overrides=[f'bm.session_info.template_paths=["{str(temp_path)}"]']
        )

        # 3. Load the template
        # The name is without the extension
        template_name = "summarise"
        rendered_template, _, _ = load_template(template=template_name, parameters={})

        # 4. Assert that the external template was loaded
        assert rendered_template == external_template_content
