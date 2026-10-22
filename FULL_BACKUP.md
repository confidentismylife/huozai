# 完整工作区压缩备份

日期：2026-10-22。当前状态：正在构建；只有最终清单和远程验证完成后才标记上传完成。

## 范围

完整快照包括现有研究目录中的代码、测试、模型权重、数据、缓存、网页、日志、预算、报告、图表、临时产物、运行依赖和实验输出。另包含用户提供的原始Excel/PPT、桌面论文目录，以及本机保存的当前项目对话与可取得的附件。

凭据文件排除，普通文件内的密钥与令牌替换为`[REDACTED]`；原有Git元数据不打包。没有删除实验负结果，也没有重新运行训练或最终测试。

## 压缩格式

完整内容保存为一个ZIP64压缩包，按45 MiB切成连续二进制分卷，每个分卷是原ZIP字节流的一部分，不能单独解压。分卷位于`backups/2026-10-22/`。

`VOLUMES.json`给出顺序、每卷长度和SHA-256，以及合并后的ZIP SHA-256；`SHA256SUMS.txt`便于使用其他工具校验。采用分卷是因为[GitHub限制普通Git单文件最大100 MiB](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)。

## 恢复

从仓库根目录运行：

```powershell
python backup_tools/restore_complete_github_backup.py --volumes backups/2026-10-22 --output restored_backup
```

恢复脚本只使用Python标准库。它依次验证每卷SHA-256、合并ZIP的SHA-256、所有ZIP成员的CRC，安全解压后再逐文件核对工作区清单。恢复目录必须是新目录，避免覆盖既有压缩包。

只合并和校验、不解压：

```powershell
python backup_tools/restore_complete_github_backup.py --volumes backups/2026-10-22 --output verification_restore --verify-only
```

## 恢复后的目录

- `workspace/`：完整研究工作区；
- `conversation/`：用户可见原始对话的Markdown、JSONL和附件；
- `provided_materials/`：用户提供的原始Excel、PPT和论文；
- `BACKUP_MANIFEST.jsonl`：工作区逐文件来源/备份大小和SHA-256；
- `SUPPLEMENT_MANIFEST.jsonl`：补充材料清单；
- `EXCLUDED_CREDENTIALS.json`：排除的凭据存储与Git元数据路径；
- `SANITIZATION_NOTES.json`：损坏压缩包测试样例等处理记录；
- `BACKUP_SUMMARY.json`：文件数、体积、脱敏次数和对话消息数。

本机绝对路径按要求保留在原始内容中。新机器运行实验前需修改环境与数据路径，并配置新的本机凭据。
